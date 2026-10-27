# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : tilt_correction.py
# Time       :2023/3/29 14:01
# Author     :Bot Zhao
# version    :python 3.9
# Description:
"""

# import libs
# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : hauff_3d.py
# Time       :2023/3/27 13:37
# Author     :Bot Zhao
# version    :python 3.9
# Description:
"""

# import libs
import SimpleITK as sitk
import numpy as np
from math import sin, cos, pi, floor, ceil
import tqdm
from utils import image_utils
import torch
import torch.nn.functional as F


def calcboundbox(input):
    """Calculate the bounding box of a set of points"""
    minx, miny, minz = np.min(input, axis=0)
    maxx, maxy, maxz = np.max(input, axis=0)
    center = [(minx + maxx) / 2, (miny + maxy) / 2, (minz + maxz) / 2]
    diag = ((maxx - minx) ** 2 + (maxy - miny) ** 2 + (maxz - minz) ** 2) ** 0.5
    return center, diag


def solve_plane(A, B, C):
    """
    Solve the plane equation.
    :param A: point A
    :param B: point B
    :param C: point C
    :return: Point (a point on the plane), Quaternion (plane quaternion), Nx (plane normal)
    """
    # Two constants
    N = np.array([0, 0, 1])
    Pi = 3.1415926535

    # Compute the unit plane normal as the cross product of BC and BA
    Nx = np.cross(B - C, B - A)
    Nx = Nx / np.linalg.norm(Nx)
    D = -np.matmul(Nx, A.T)

    # Compute the unit rotation axis and rotation angle in [0, pi]
    Nv = np.cross(Nx, N)
    angle = np.arccos(np.dot(Nx, N))

    # Handle the assumption that the angle between the two vectors does not exceed pi/2
    if angle > Pi / 2.0:
        angle = Pi - angle
        Nv = -Nv

    # Point = (A + B + C) / 3.0
    Point = B
    # Compute the unit quaternion
    Quaternion = np.append(Nv * sin(angle / 2), cos(angle / 2))
    # print("Rotation axis:\t", Nv)
    # print("Rotation angle:\t", angle)
    # print("Corresponding quaternion:\t", Quaternion)

    return Point, Quaternion, np.concatenate([Nv, np.array(D)[np.newaxis]])


def solve_distance(M, P, N):
    """
    Compute the distance from point M to plane (P, Q).
    :param M: point M
    :param P: a point on the plane
    :param N: plane normal
    :return: distance from the point to the plane
    """

    # Convert the quaternion to a normal vector
    # A = 2 * Q[0] * Q[2] + 2 * Q[1] * Q[3]
    # B = 2 * Q[1] * Q[2] - 2 * Q[0] * Q[3]
    # C = -Q[0] ** 2 - Q[1] ** 2 + Q[2] ** 2 + Q[3] ** 2
    # D = -A * P[0] - B * P[1] - C * P[2]

    # Use the solved normal vector directly for simpler computation
    A = N[0]
    B = N[1]
    C = N[2]
    D = -A * P[0] - B * P[1] - C * P[2]

    return np.fabs(A * M[0] + B * M[1] + C * M[2] + D) / np.sqrt(A ** 2 + B ** 2 + C ** 2)


def RANSAC(data):
    """
    Estimate the model with RANSAC.
    """
    # Dataset size
    SIZE = data.shape[0]
    # maximum iterations; improved estimates update iters, default: 10000
    iters = 10000
    # accepted residual between data and model, default: 0.25
    sigma = 0.25
    # number of inliers
    pretotal = 0
    # desired probability of finding the correct model, default: 0.99
    Per = 0.999
    # Initialize
    P = np.array([])
    Q = np.array([])
    N = np.array([])
    for i in range(iters):
        # Randomly select three data points to solve the model
        sample_index = np.random.choice(range(SIZE), 3)
        P, Q, N = solve_plane(data[sample_index[0]], data[sample_index[1]], data[sample_index[2]])

        # Count inliers
        total_inlier = 0
        for index in range(SIZE):
            if solve_distance(data[index], P, N) < sigma:
                total_inlier = total_inlier + 1

        print("total_inlier:{}/{} ".format(total_inlier, SIZE))
        # Check whether the current model improves on the previous estimate
        if total_inlier > pretotal:
            iters = np.log(1 - Per) / np.log(1 - pow(total_inlier / SIZE, 2))
            pretotal = total_inlier

        # Check whether the current model fits more than half of the points
        if total_inlier > SIZE / 2:
            break
    return P, Q, N, total_inlier


class Tilt_Correction(object):
    def __init__(self, img_path, new_spacing, gaussion_sigma=0.,
                 drop_x=None, drop_y=None, drop_z=None, edge_range_min=0, edge_range_max=None,
                 theta_start=-pi / 18, theta_end=pi / 18, phi_start=-pi / 18, phi_end=pi / 18, anglestep=pi / 180):
        self.edge_range_min = edge_range_min
        self.edge_range_max = edge_range_max
        self.new_spacing = new_spacing
        self.img_path = img_path
        self.gaussion_sigma = gaussion_sigma
        self.drop_x = drop_x
        self.drop_y = drop_y
        self.drop_z = drop_z
        self.img = sitk.ReadImage(img_path)
        self.theta_start = theta_start
        self.theta_end = theta_end
        self.phi_start = phi_start
        self.phi_end = phi_end
        self.anglestep = anglestep

        self.phi = None
        self.theta = None
        self.center = None
        self.ds_edge = None
        self.raw_edge = None
        self.Direction = None
        self.Origin = None
        self.spacing = None
        self.plane_param = None
        self.shape = None
        self.points = None

    def get_edge(self, save_path=None):
        # Read the image
        image = self.img
        image_arr = sitk.GetArrayFromImage(image)
        image_arr = image_arr[self.drop_z[0]:self.drop_z[1], self.drop_y[0]:self.drop_y[1],
                    self.drop_x[0]:self.drop_x[1]]
        image_arr = (image_arr - np.min(image_arr)) / (np.max(image_arr) - np.min(image_arr))

        crop_image = sitk.GetImageFromArray(image_arr * 255)
        crop_image.SetOrigin(image.GetOrigin())
        crop_image.SetSpacing(image.GetSpacing())
        crop_image.SetDirection(image.GetDirection())

        self.spacing = image.GetSpacing()
        self.Origin = image.GetOrigin()
        self.Direction = image.GetDirection()

        # Gaussian filtering
        gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian.SetSigma(self.gaussion_sigma * self.spacing[0])
        image = gaussian.Execute(crop_image)

        # Edge detection
        crop_image = sitk.Cast(image, sitk.sitkFloat32)
        edge_filter = sitk.SobelEdgeDetectionImageFilter()
        edge_image = edge_filter.Execute(crop_image)

        # Downsample
        if self.new_spacing is not None:
            ds_image = image_utils.sample_2_fixed_spacing(edge_image, self.new_spacing)
            image_gauss = image_utils.sample_2_fixed_spacing(image, self.new_spacing)
            sitk.WriteImage(image_gauss, self.img_path.replace('.nii', '_gauss.nii'))
        else:
            ds_image = image
        edge_arr = sitk.GetArrayFromImage(ds_image)
        print("==>Down-sampled arr size:", edge_arr.shape)

        # Save the output
        edge_arr[edge_arr < self.edge_range_min] = 0
        if self.edge_range_max is not None:
            edge_arr[edge_arr > self.edge_range_max] = 0
        edge_arr[edge_arr != 0] = 1
        self.ds_edge = edge_arr.copy()

        #
        edge_arr = np.transpose(edge_arr, (2, 1, 0))  # x, y, z index order
        points = np.nonzero(edge_arr)
        self.points = np.stack([points[0], points[1], points[2]], axis=1)
        self.points = self.points * np.array(self.new_spacing)  # convert to physical coordinates
        print("==>Edge Points: ", len(self.points))

        if save_path is not None:
            edge_image = sitk.GetImageFromArray(self.ds_edge)
            edge_image.SetOrigin(self.Origin)
            edge_image.SetSpacing((self.new_spacing))
            edge_image.SetDirection(self.Direction)
            sitk.WriteImage(edge_image, save_path)

    def hough_transform(self):
        n = self.points.shape[0]
        if n < 3:
            return None, None, None, None
        theta_start, theta_end = self.theta_start, self.theta_end
        phi_start, phi_end = self.phi_start, self.phi_end
        # phi_start, phi_end = 0.25*pi, 0.75*pi
        anglestep = self.anglestep
        disstep = ((self.new_spacing[0] ** 2 + self.new_spacing[1] ** 2 + self.new_spacing[2] ** 2) ** 0.5)

        ptCenter, diag = calcboundbox(self.points)
        d_start, d_end = -diag / 2.0, diag / 2.0
        print("==>Center coordinates:", ptCenter)

        thetas = int(ceil((theta_end - theta_start) / anglestep))
        phis = int(ceil((phi_end - phi_start) / anglestep))
        dises = int(ceil(diag / disstep))

        cube = np.zeros((thetas, phis, dises), dtype=np.int32)

        print("==>Hough transform: populate the accumulator", cube.shape)
        for i in tqdm.tqdm(range(n)):
            ptOrigin = self.points[i]
            point = ptOrigin - ptCenter
            theta = theta_start
            for j in range(thetas):
                phi = phi_start
                for k in range(phis):
                    sinphi = sin(phi)
                    d = cos(theta) * sinphi * point[0] + sin(theta) * sinphi * point[1] + cos(phi) * point[2]
                    d_index = int(floor((d - d_start) / disstep))
                    cube[j, k, d_index] += 1
                    phi += anglestep
                    if phi > phi_end:
                        break
                theta += anglestep
                if theta > theta_end:
                    break

        # buf = 1
        # maxcount = 0
        # xmax, ymax, zmax = 0, 0, 0

        # for i in range(thetas):
        #     for j in range(phis):
        #         for k in range(buf, dises - buf):
        #             count = 0
        #             for x in range(i - buf, i + buf + 1):
        #                 for y in range(j - buf, j + buf + 1):
        #                     for z in range(k - buf, k + buf + 1):
        #                         count += cube[x % thetas, y % phis, z]
        #             if count > maxcount:
        #                 xmax, ymax, zmax = i, j, k
        #                 maxcount = count
        xmax, ymax, zmax = np.where(cube == np.max(cube))
        print("===>max cube number is {} at {},{},{}".format(np.max(cube), xmax, ymax, zmax))
        xmax, ymax, zmax = xmax[-1].item(), ymax[0].item(), zmax[0].item()

        theta = theta_start + xmax * anglestep
        phi = phi_start + ymax * anglestep
        d = d_start + zmax * disstep
        print('===>theta:{},phi:{},d:{}'.format(theta, phi, d))
        # theta, phi = phi, theta

        A = cos(theta) * sin(phi)
        B = sin(theta) * sin(phi)
        C = cos(phi)
        D = -d - (A * ptCenter[0] + B * ptCenter[1] + C * ptCenter[2])
        self.plane_param = [A, B, C, D]
        self.center = ptCenter
        self.theta = theta
        self.phi = phi
        print('===>A:{},B:{},C:{},D:{}'.format(A, B, C, D))
        # return A, B, C, D

    def rassac(self):
        ptCenter, diag = calcboundbox(self.points)
        points = self.points - ptCenter
        P, Q, N = RANSAC(points)
        phi = np.arccos(N[-1])
        theta = np.arcsin(N[1] / sin(phi))
        self.center = ptCenter
        self.plane_param = Q
        print(P, Q, N)
        print(phi, theta)

    def draw_plane(self, save_path=None):
        A, B, C, D = self.plane_param
        plane = np.zeros_like(self.ds_edge)
        z, y, x = plane.shape  # shape: z, y, x

        xx = np.arange(0, x)
        yy = np.arange(0, y)
        Y, X = np.meshgrid(yy, xx)
        X = X * self.new_spacing[0] - self.center[0]
        Y = Y * self.new_spacing[1] - self.center[1]
        Z = (-D - A * X - B * Y) / C
        Z = (Z + self.center[2])
        plane = np.transpose(plane, (2, 1, 0))
        print(plane.shape)
        for i in xx:
            for j in yy:
                z = int(Z[i, j] / self.new_spacing[2])
                try:
                    plane[i, j, z] = 1
                except:
                    print(i, j, z, 'is overpass the region')
        plane = np.transpose(plane, (2, 1, 0))
        if save_path is not None:
            image = sitk.GetImageFromArray(plane)
            image.SetOrigin(self.Origin)
            image.SetSpacing(self.new_spacing)
            image.SetDirection(self.Direction)
            sitk.WriteImage(image, save_path)

    def tilt_correction(self, img_path):
        """
        tilt correction by the fitted incline plane (controlled by theta and phi)
        """
        img = sitk.ReadImage(self.img_path)
        img_arr = sitk.GetArrayFromImage(img)
        img_arr = np.transpose(img_arr, (2, 1, 0))  # x, y, z index order
        img_tensor = torch.from_numpy(img_arr[np.newaxis, np.newaxis, ...])
        shear_x = torch.eye(4)
        shear_y = torch.eye(4)
        shear_z = torch.eye(4)

        rotate_y, rotate_z = torch.Tensor([-self.theta]), torch.Tensor([-self.phi])
        shear_x[0, 2] = torch.tan(rotate_y)
        shear_y[0, 1] = torch.tan(rotate_z)

        shear_m = torch.mm(shear_x, torch.mm(shear_y, shear_z))
        final_matrix = shear_m
        grid = F.affine_grid(final_matrix[0:3].unsqueeze(0), img_tensor.size(), align_corners=False)
        transformed_x = F.grid_sample(img_tensor, grid, mode='bilinear', align_corners=False)
        transformed_x = transformed_x.squeeze().numpy()

        transformed_x = np.transpose(transformed_x, (2, 1, 0))  # x, y, z index order
        img_corr = sitk.GetImageFromArray(transformed_x)
        img_corr.SetOrigin(img.GetOrigin())
        img_corr.SetSpacing(img.GetSpacing())
        img_corr.SetDirection(img.GetDirection())
        sitk.WriteImage(img_corr, self.img_path.replace('.nii', '_corr.nii'))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Tilt Correction by Hough-Transform')
    parser.add_argument('-img_path', '--img_path',
                        required=True,
                        help='input file')
    parser.add_argument('-gs', '--gaussion_sigma', default=1.0,
                        help='Gaussion_sigma')
    parser.add_argument('-ds_spacing', '--new_spacing', default=(0.06, 0.06, 0.02),
                        help='The new spacing of down-sample for acc')
    parser.add_argument('-drop_x', '--drop_x', default=[60, -60],
                        help='The cropping range at X direction for selecting the edge ROI')
    parser.add_argument('-drop_y', '--drop_y', default=[60, -60],
                        help='The cropping range at Y direction for selecting the edge ROI')
    parser.add_argument('-drop_z', '--drop_z', default=[10, 50],
                        help='The cropping range at Z direction for selecting the edge ROI')
    parser.add_argument('-grad_range', '--grad_range', default=[500, None],
                        help='The gradient value range for the edge detection.')
    parser.add_argument('-theta_range', '--theta_range', default=[-pi / 18, pi / 18],
                        help='The theta value range for Haugh-transform.')
    parser.add_argument('-phi_range', '--phi_range', default=[-pi / 18, pi / 18],
                        help='The phi value range for Haugh-transform.')
    parser.add_argument('-anglestep', '--anglestep', default=pi / 180,
                        help='The step length of the angle.')
    parser.add_argument('-save_edge', '--save_edge', default='_edge.nii',
                        help='The step length of the angle.')
    parser.add_argument('-save_plane', '--save_plane', default='_detected_plane2.nii',
                        help='The step length of the angle.')
    arguments = parser.parse_args()

    path = arguments.img_path
    gaussion_sigma = arguments.gaussion_sigma
    new_spacing = arguments.new_spacing
    drop_x = arguments.drop_x
    drop_y = arguments.drop_y
    drop_z = arguments.drop_z
    edge_range_min = arguments.grad_range[0]
    edge_range_max = arguments.grad_range[1]
    theta_range = arguments.theta_range
    phi_range = arguments.phi_range
    anglestep = arguments.anglestep
    save_edge = path.replace('.nii', arguments.save_edge)
    save_plane = path.replace('.nii', arguments.save_plane)

    hauff_transform = Tilt_Correction(path, new_spacing, gaussion_sigma=gaussion_sigma,
                                      drop_x=drop_x, drop_y=drop_y, drop_z=drop_z,
                                      edge_range_min=edge_range_min, edge_range_max=edge_range_max,
                                      theta_start=theta_range[0], theta_end=theta_range[1], phi_start=phi_range[0],
                                      phi_end=phi_range[1], anglestep=anglestep)

    hauff_transform.get_edge(save_path=save_edge)

    hauff_transform.hough_transform()
    # hauff_transform.rassac()

    hauff_transform.draw_plane(save_path=save_plane)
    hauff_transform.tilt_correction(path)

    print('Done!')
