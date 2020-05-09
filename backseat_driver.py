'''
Implements a "backseat driver" that uses semantic segmentation and depth data
to monitor the safety of the car's planned trajectory in real-time.

Written by Charles Dawson (cbd@mit.edu) on May 3, 2020, for the 16.412 Grand
Challenge, on the Semantic Segmentation team.
'''
import glob
import os
import sys
import copy
import time

import numpy as np

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from image_converter import (
    depth_to_local_point_cloud, to_rgb_array,
    labels_to_cityscapes_palette
)

from collision_detection import get_distance


class BackseatDriver:
    '''The BackseatDriver collects semantic segmentation, depth, and
    planned trajectory data at whatever rate it is published, then exposes a
    callback for querying its own estimate of the safety of planned
    trajectories (i.e. if the planned trajectory will result in a future 
    collision and if so, how long until that predicted collision) at a constant
    rate.

    Each BackseatDriver will provide two functions (`depth_callback` and
    `semantic_segmentation_callback`) that should be used as callbacks and
    called on new depth and semantic segmentation images whenever those
    images become available. BackseatDriver will save these data and use
    the most recent frame for which both images are available to generate
    its collision warnings.

    In addition to these two callbacks, BackseatDriver exposes the
    `update_planned_trajectory` function, which should be called to inform the
    BackseatDriver of the intended trajectory. This trajectory (expressed in
    the ego-vehicle frame) will be checked for collision against the segmented
    point cloud generated with data supplied to the two callbacks described
    above.

    Finally, the BackseatDriver exposes a callback `get_safety_estimate` that
    can be registered with `carla.World.on_tick`. When called, the
    `get_safety_estimate` function will print a safety report and return the
    distance until collision (which might be `float('inf')` if no collision is
    forseen).
    '''

    def __init__(self,
                 camera_transform,
                 hazard_labels,
                 image_size,
                 update_rate=10,
                 horizon=0.2,
                 debug=False):
        '''Initializes the BackseatDriver to publish safety reports
        at the specified update_rate (in Hz). Once initialized, the member
        function start() must be called to start the asynchronous safety-
        monitoring loop.

        @param camera_transform: the carla.Transform representing the
                                 pose of the camera in the ego vehicle frame.
        @param hazard_labels: a list of integers describing semantic
                              segmentation labels to avoid hitting (e.g.
                              other cars, pedestrians).
        @param horizon: the maximum distance from the camera that will checked
                        for collision (in km). This saves lots of computation
                        by avoiding checking for collision with the sky.
                        Defaults to 0.2 km (200 m).
        @param debug: set to True to enable debug logging.
        '''
        # Since we receive data from semantic segmentation and depth cameras
        # asynchronously, we need to create dictionaries to store the images
        # (indexed by frame)
        self.depth_data = dict()
        self.semantic_data = dict()

        # Initialize a place to store the trajectory
        self.trajectory = None

        # Save the transform from the vehicle to the camera
        self.camera_transform = camera_transform
        # Save the hazard labels
        self.hazard_labels = hazard_labels
        # Save the horizon
        self.horizon = horizon
        # Save debug status
        self.debug = debug

    def log(self, message):
        '''Logs a message to the console if debug logging is enabled.

        @param message: the string message to print
        '''
        # Do not print if debug is not enabled.
        if not self.debug:
            return

        # Otherwise, add a timestamp to the message and print it
        print("[BackseatDriver]" +
              time.strftime("[%a, %d %b %Y %H:%M:%S]: ", time.localtime()) +
              message)

    def depth_callback(self, depth_data):
        '''Receives depth data asynchronously and saves it.

        @param depth_data: a carla.Image object containing a depth image.
                           These data should be encoded as an 18-bit number,
                           with the 8 least-significant bits in the red,
                           channel, the next 8 least-significant bits in the
                           green channel, and the 8 most-significant bits in
                           the blue channel. The depth should measure distance
                           perpendicular to the camera plane (note: this is
                           different from Euclidean distance).
        '''
        # Save the depth data along with its frame
        self.depth_data[depth_data.frame] = depth_data
        self.log("Received depth data for frame: " + str(depth_data.frame))

    def semantic_segmentation_callback(self, semantic_data):
        '''Receives semantic_segmentation data asynchronously and saves it.

        @param semantic_data:
            a dict containing keys `original` for the original carla.Image
            containing an RGB picture of the road, and `segmented` containing
            a numpy array with an integer label for each pixel.
        TODO: Confirm label encoding with Lars
        '''
        # Save the depth data along with its frame
        self.semantic_data[semantic_data['original'].frame] = semantic_data
        self.log("Received semantic data for frame: " +
                 str(semantic_data.frame))

    def update_planned_trajectory(self, trajectory):
        ''' "Files a flight plan" by telling the backseat driver what
        trajectory the car plans to follow in the near future.

        Following the common convention, the trajectory should be expressed
        in the ego vehicle frame.

        @param trajectory: an Nx4 numpy array, where each row is (t, x, y,
                           theta, alpha) denoting a time-indexed list of
                           trajectory waypoints, where
            - t is the time of the waypoint.
            - x, y, theta denotes the 2D pose of the vehicle at this waypoint,
              where (x, y, theta) = (0, 0, 0) is the current location of the
              vehicle (with the x-axis pointing along the current driving
              direction of the car).
            - alpha is the steering angle of the car at the specified waypoint.
        '''
        self.trajectory = trajectory
        self.log("Received trajectory with " + str(len(self.trajectory)) +
                 "waypoints")

    def get_safety_estimate(self, world_snapshot):
        ''' Checks for collision between the planned trajectory
        and the environment, as perceived via a depth camera and semantic
        segmentation.

        Should be run via carla.World.on_tick.

        Relies on data gathered by the depth_callback,
        semantic_segmentation_callback, and update_planned_trajectory
        functions.

        @param world_snapshot: a carla.WorldSnapshot provided by the on_tick
                               callback manager.
        '''
        # We have to start by getting the most recent frame for which both
        # semantic segmentation and depth data is stored. Recall that we've
        # stored these data indexed by integer frame numbers, so we want
        # the highest frame number

        # Sort depth frame numbers in descending order
        depth_frame_numbers = sorted(list(self.depth_data.keys()),
                                     reverse=True)
        # Then scoot down the list until we find one with matching semantic
        # segmentation data
        matching_frame_found = False
        for frame_number in depth_frame_numbers:
            if frame_number in self.semantic_data:
                matching_frame_found = True
                break
        # If we didn't find a matching frame, we can't do anything
        if not matching_frame_found:
            self.log("No matching frames found. Can't make a safety estimate!")
            return

        # Otherwise, we can generate the safety estimate
        if matching_frame_found:
            # Get the depth and semantic data from storage, and extract a
            # timestamp and other useful metadata for this frame
            depth_data = self.depth_data[frame_number]
            semantic_data = self.semantic_data[frame_number]
            # get the timestamp for these data and metadata
            timestamp = depth_data.timestamp

            # To save memory, now clear both storage dictionaries of any
            # frames less than the current frame number
            for frame_to_delete in self.depth_data.keys():
                if frame_to_delete < frame_number:
                    self.depth_data.pop(frame_to_delete, None)
            for frame_to_delete in self.semantic_data.keys():
                if frame_to_delete < frame_number:
                    self.depth_data.pop(frame_to_delete, None)

            # At this point, we have the depth and semantic data that we
            # want to fuse into a segmented point cloud.

            #   1) Convert the segmentation data to a labelled array
            #        (this should be a width x height x rgb array of floats,
            #         where the label of each pixel is encoded in the red
            #         value of each pixel)
            semantic_labels = to_rgb_array(semantic_data)

            #   2) Create a point cloud that contains only points
            #      that we labelled as hazards
            self.log("Making point cloud for frame " + str(frame_number))
            point_cloud = depth_to_local_point_cloud(
                depth_data,
                semantic_labels,
                max_depth=self.max_depth
            )

            # We want to check the trajectory (starting at the current time)
            # for collision. Skip if no waypoints left
            # Iterate through the rows of self.trajectory to find the first
            # waypoint with after the current timestamp
            start_index = None
            for i in range(self.trajectory.shape[0]):
                if self.trajectory[i, 0] >= timestamp:
                    start_index = i
                    break
            if not start_index:
                self.log(("No trajectory waypoints left."
                          "Cannot generate safety report!"))

            # Now iterate through the remaining waypoints
            for i in range(start_index, self.trajectory.shape[0]):
                # The collision checking code takes in one pose and a list of
                # points in the waypoint frame, and it returns the distance
                # until collision. However, we can consider a waypoint to be
                # collision-free if the distance to collision is more than the
                # distance to the next waypoint.

                # To do the collision checking, we first need to
                # transform the point cloud into the waypoint frame
                x = self.trajectory[i, 1]
                y = self.trajectory[i, 2]
                theta = self.trajectory[i, 3]
                alpha = self.trajectory[i, 4]
                points = point_cloud.offset_then_rotate(x, y, theta)

                # Call Shangjie's code for collision checking
                distance_to_collision = get_distance(alpha, points)

                # If collision occurs before next waypoint, raise the alarm
                if i < self.trajectory.shape[0] - 1:
                    next_x = self.trajectory[i + 1, 1]
                    next_y = self.trajectory[i + 1, 2]
                    distance_to_next_waypoint = np.sqrt((next_x - x) ** 2 +
                                                        (next_y - y) ** 2)

                    if distance_to_collision < distance_to_next_waypoint:
                        self.log("WARNING: Collision predicted. Distance: " +
                                 str(distance_to_collision))
                        break

                else:  # if we're on the last waypoint, raise the alarm
                    self.log("WARNING: Collision predicted. Distance: " +
                             str(distance_to_collision))

                #  Otherwise, continue to next waypoint.

        # Nothing to return right now.