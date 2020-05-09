# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma
# de Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Modifications to depth_to_local_point_cloud and PointCloud.offset_all_points
# were made by Charles Dawson (cbd@mit.edu) and
# are also released under the MIT license

"""
Handy conversions for CARLA images.

The functions here are provided for real-time display, if you want to save the
converted images, save the images from Python without conversion and convert
them afterwards with the C++ implementation at "Util/ImageConverter" as it
provides considerably better performance.
"""

import math

try:
    import numpy
    from numpy.matlib import repmat
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

import glob
import os
import sys

from collections import namedtuple

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


# ==============================================================================
# -- Helpers ------------------------------------------------------------------
# ==============================================================================


Color = namedtuple('Color', 'r g b')
Color.__new__.__defaults__ = (0, 0, 0)


Point = namedtuple('Point', 'x y z color')
Point.__new__.__defaults__ = (0.0, 0.0, 0.0, None)


def _append_extension(filename, ext):
    return filename if filename.lower().endswith(ext.lower()) \
        else filename + ext


class SensorData(object):
    """Base class for sensor data returned from the server."""
    def __init__(self, frame_number):
        self.frame_number = frame_number


class PointCloud(SensorData):
    """A list of points."""

    def __init__(self, frame_number, array, color_array=None):
        super(PointCloud, self).__init__(frame_number)
        self._array = array
        self._color_array = color_array
        self._has_colors = color_array is not None

    @property
    def array(self):
        """The numpy array holding the point-cloud.
        3D points format for n elements:
        [ [X0,Y0,Z0],
          ...,
          [Xn,Yn,Zn] ]
        """
        return self._array

    @property
    def color_array(self):
        """The numpy array holding the colors corresponding to each point.
        It is None if there are no colors.
        Colors format for n elements:
        [ [R0,G0,B0],
          ...,
          [Rn,Gn,Bn] ]
        """
        return self._color_array

    def has_colors(self):
        """Return whether the points have color."""
        return self._has_colors

    def apply_transform(self, transformation):
        """Modify the PointCloud instance transforming its points"""
        self._array = transformation.transform_points(self._array)

    def offset_then_rotate(self, x, y, theta):
        """Offset all points so that the point (x, y) -> (0, 0), then rotate
        by theta around the +z axis.

        Does not modify this point cloud, but returns the transformed points.
        """
        offset_points = self._array - [x, y, 0]
        rotation_matrix = numpy.array(
            [[numpy.cos(theta), -numpy.sin(theta), 0],
             [numpy.sin(theta), numpy.cos(theta),  0],
             [0,                0,                 1]])
        return rotation_matrix.dot(offset_points.T)

    def save_to_disk(self, filename):
        """Save this point-cloud to disk as PLY format."""
        filename = _append_extension(filename, '.ply')

        def construct_ply_header():
            """Generates a PLY header given a total number of 3D points and
            coloring property if specified
            """
            points = len(self)  # Total point number
            header = ['ply',
                      'format ascii 1.0',
                      'element vertex {}',
                      'property float32 x',
                      'property float32 y',
                      'property float32 z',
                      'property uchar diffuse_red',
                      'property uchar diffuse_green',
                      'property uchar diffuse_blue',
                      'end_header']
            if not self._has_colors:
                return '\n'.join(header[0:6] + [header[-1]]).format(points)
            return '\n'.join(header).format(points)

        if not self._has_colors:
            ply = '\n'.join(['{:.2f} {:.2f} {:.2f}'.format(
                *p) for p in self._array.tolist()])
        else:
            points_3d = numpy.concatenate(
                (self._array, self._color_array), axis=1)
            ply = '\n'.join(['{:.2f} {:.2f} {:.2f} {:.0f} {:.0f} {:.0f}'
                             .format(*p) for p in points_3d.tolist()])

        # Create folder to save if does not exist.
        folder = os.path.dirname(filename)
        if not os.path.isdir(folder):
            os.makedirs(folder)

        # Open the file and save with the specific PLY format.
        with open(filename, 'w+') as ply_file:
            ply_file.write('\n'.join([construct_ply_header(), ply]))

    def __len__(self):
        return len(self.array)

    def __getitem__(self, key):
        color = None if self._color_array is None else Color(
            *self._color_array[key])
        return Point(*self._array[key], color=color)

    def __iter__(self):
        class PointIterator(object):
            """Iterator class for PointCloud"""

            def __init__(self, point_cloud):
                self.point_cloud = point_cloud
                self.index = -1

            def __next__(self):
                self.index += 1
                if self.index >= len(self.point_cloud):
                    raise StopIteration
                return self.point_cloud[self.index]

            def next(self):
                return self.__next__()

        return PointIterator(self)

    def __str__(self):
        return str(self.array)


def to_bgra_array(image):
    """Convert a CARLA raw image to a BGRA numpy array."""
    if not isinstance(image, carla.Image):
        raise ValueError("Argument must be a carla.sensor.Image")
    array = numpy.frombuffer(image.raw_data, dtype=numpy.dtype("uint8"))
    array = numpy.reshape(array, (image.height, image.width, 4))
    return array


def to_rgb_array(image):
    """Convert a CARLA raw image to a RGB numpy array."""
    array = to_bgra_array(image)
    # Convert BGRA to RGB.
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array


def labels_to_array(image):
    """
    Convert an image containing CARLA semantic segmentation labels to a
    2D array containing the label of each pixel.
    """
    return to_bgra_array(image)[:, :, 2]


def labels_to_cityscapes_palette(image):
    """
    Convert an image containing CARLA semantic segmentation labels to
    Cityscapes palette.
    """
    classes = {
        0: [0, 0, 0],         # None
        1: [70, 70, 70],      # Buildings
        2: [190, 153, 153],   # Fences
        3: [72, 0, 90],       # Other
        4: [220, 20, 60],     # Pedestrians
        5: [153, 153, 153],   # Poles
        6: [157, 234, 50],    # RoadLines
        7: [128, 64, 128],    # Roads
        8: [244, 35, 232],    # Sidewalks
        9: [107, 142, 35],    # Vegetation
        10: [0, 0, 255],      # Vehicles
        11: [102, 102, 156],  # Walls
        12: [220, 220, 0]     # TrafficSigns
    }
    array = labels_to_array(image)
    result = numpy.zeros((array.shape[0], array.shape[1], 3))
    for key, value in classes.items():
        result[numpy.where(array == key)] = value
    return result


def depth_to_array(image):
    """
    Convert an image containing CARLA encoded depth-map to a 2D array
    containing the depth value of each pixel normalized between [0.0, 1.0].
    """
    array = to_bgra_array(image)
    array = array.astype(numpy.float32)
    # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
    normalized_depth = numpy.dot(array[:, :, :3], [65536.0, 256.0, 1.0])
    normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
    return normalized_depth


def depth_to_logarithmic_grayscale(image):
    """
    Convert an image containing CARLA encoded depth-map to a logarithmic
    grayscale image array.
    "max_depth" is used to omit the points that are far enough.
    """
    normalized_depth = depth_to_array(image)
    # Convert to logarithmic depth.
    logdepth = numpy.ones(normalized_depth.shape) + \
        (numpy.log(normalized_depth) / 5.70378)
    logdepth = numpy.clip(logdepth, 0.0, 1.0)
    logdepth *= 255.0
    # Expand to three colors.
    return numpy.repeat(logdepth[:, :, numpy.newaxis], 3, axis=2)


def depth_to_local_point_cloud(image, color=None, max_depth=0.9,
                               sampling_rate=1,
                               hazard_labels=set([1, 4, 5, 10, 11])):
    """
    Convert an image containing CARLA encoded depth-map to a 2D array
    containing the 3D position (relative to the camera) of each pixel
    and its corresponding RGB color of an array.
    "max_depth" is used to omit the points that are far enough.

    @param sampling_rate: an integer indicating how much to downscale the
                          image and color map. 1 keeps the same resolution,
                          higher numbers reduce the resolution. Must be
                          greater than or equal to 1
    @param hazard_labels: a list of integers that are marked as hazards. The
                          returned point cloud will only contain hazard points.
    """
    far = 1000.0  # max depth in meters.
    normalized_depth = depth_to_array(image)

    # resize images
    assert(sampling_rate >= 1)
    if sampling_rate > 1:
        normalized_depth = normalized_depth[::sampling_rate, ::sampling_rate]
        color = color[::sampling_rate, ::sampling_rate, :]

    width = normalized_depth.shape[1]
    height = normalized_depth.shape[0]

    # (Intrinsic) K Matrix
    k = numpy.identity(3)
    k[0, 2] = width / 2.0
    k[1, 2] = height / 2.0
    k[0, 0] = k[1, 1] = width / \
        (2.0 * math.tan(image.fov * math.pi / 360.0))

    # 2d pixel coordinates
    pixel_length = width * height
    u_coord = repmat(numpy.r_[width - 1:-1:-1],
                     height, 1).reshape(pixel_length)
    v_coord = repmat(numpy.c_[height - 1:-1:-1],
                     1, width).reshape(pixel_length)
    if color is not None:
        color = color.reshape(pixel_length, 3)
    normalized_depth = numpy.reshape(normalized_depth, pixel_length)

    # Search for pixels where the depth is greater than max_depth to
    # delete them
    max_depth_indexes = numpy.where(normalized_depth > max_depth)
    normalized_depth = numpy.delete(normalized_depth, max_depth_indexes)
    u_coord = numpy.delete(u_coord, max_depth_indexes)
    v_coord = numpy.delete(v_coord, max_depth_indexes)
    if color is not None:
        color = numpy.delete(color, max_depth_indexes, axis=0)

        # Also delete pixels that are not marked as hazards
        if hazard_labels:
            not_hazards = numpy.isin(color[:, :, 0],
                                     hazard_labels,
                                     invert=True)
            color = numpy.delete(color, not_hazards, axis=0)
            normalized_depth = numpy.delete(normalized_depth, not_hazards)
            u_coord = numpy.delete(u_coord, not_hazards)
            v_coord = numpy.delete(v_coord, not_hazards)

    # pd2 = [u,v,1]
    p2d = numpy.array([u_coord, v_coord, numpy.ones_like(u_coord)])

    # P = [X,Y,Z]
    p3d = numpy.dot(numpy.linalg.inv(k), p2d)
    p3d *= normalized_depth * far

    # Formating the output to:
    # [[X1,Y1,Z1,R1,G1,B1],[X2,Y2,Z2,R2,G2,B2], ... [Xn,Yn,Zn,Rn,Gn,Bn]]
    if color is not None:
        # numpy.concatenate((numpy.transpose(p3d), color), axis=1)
        return PointCloud(
            image.frame_number,
            numpy.transpose(p3d),
            color_array=color)
    # [[X1,Y1,Z1],[X2,Y2,Z2], ... [Xn,Yn,Zn]]
    return PointCloud(image.frame_number, numpy.transpose(p3d))
