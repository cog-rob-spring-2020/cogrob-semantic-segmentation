import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from .image_converter import (
    depth_to_local_point_cloud, labels_to_array,
    labels_to_cityscapes_palette
)


class PointCloudSynthesizer:
    '''An object that collects semantic segmentation and depth images
    and fuses them into a 3D segmented point cloud
    '''
    def __init__(self, camera_transform):
        # We receive frames from the depth camera and semantic segmentation
        # view asynchronously. We save the data for each frame in these dicts,
        # and once we have both pieces of data for a frame we'll spawn a thread
        # to fuse the data and save it.
        #
        # Both dicts will be indexed by frame
        self.depth_data = dict()
        self.semantic_data = dict()

        # Maximum depth to include in point cloud (so that sky is excluded)
        self.max_depth = 0.2  # km

        # Save the transform of the camera
        self.camera_transform = camera_transform

        # The directory in which to save the point clouds
        self.output_folder = 'tutorial_pt_cloud/ply'

    def depth_callback(self, depth_data):
        '''Receives depth data asynchronously, and spawns a thread to
        fuse the depth and semantic data if it has both depth and semantics
        for the current frame.'''

        # Save the depth data along with its frame
        self.depth_data[depth_data.frame] = depth_data

        # If we have the matching semantic data, fuse
        if depth_data.frame in self.semantic_data:
            self.fuse(depth_data.frame)

    def semantic_callback(self, semantic_data):
        '''Receives semantic data asynchronously, and fuses
        the depth and semantic data if it has both depth and semantics
        for the current frame.'''

        # Save the semantic data along with its frame
        self.semantic_data[semantic_data.frame] = semantic_data

        # If we have the matching semantic data, fuse
        if semantic_data.frame in self.depth_data:
            self.fuse(semantic_data.frame)

    def fuse(self, frame):
        '''Fuses the depth and semantic data for this frame.
        '''
        # Get the depth and semantic data, removing them from the dict in
        # the process (to save space)
        depth_data = self.depth_data.pop(frame)
        semantic_data = self.semantic_data.pop(frame)

        # Convert the semantic image to an numpy array
        # semantic_labels = labels_to_array(semantic_data)
        semantic_rgb = labels_to_cityscapes_palette(semantic_data)

        point_cloud = depth_to_local_point_cloud(
            depth_data,
            semantic_rgb,
            max_depth=self.max_depth
        )

        # Transform the point cloud into world coordinates
        # Save point cloud to disk
        # Save PLY to disk
        # This generates the PLY string with the 3D points and the RGB colors
        # for each row of the file.
        point_cloud.save_to_disk(os.path.join(
            self.output_folder, '{:0>5}.ply'.format(frame))
        )
