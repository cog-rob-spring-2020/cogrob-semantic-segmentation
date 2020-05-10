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

sys.path += [
  "/opt/carla/PythonAPI/carla_scripts/light-weight-refinenet"
]
from RefineNet import RefineNet


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

from collision_detection import get_collision


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
                 hazard_labels=[1, 4, 5, 10, 11],
                 horizon=0.2,
                 update_rate=10,
                 debug=False):
        '''Initializes the BackseatDriver to provide safety reports.
        Once initialized, the member function get_safety_estimate must be
        called by the user to get safety estimates, and the depth_callback,
        semantic_segmentation_callback, and update_planned_trajectory callbacks
        must be called to provide the BackseatDriver with the requisite data.

        @param camera_transform: the carla.Transform representing the
                                 pose of the camera in the ego vehicle frame.
        @param hazard_labels: a list of integers describing semantic
                              segmentation labels to avoid hitting (e.g.
                              other cars, pedestrians). Defaults to
                              [1, 4, 5, 10, 11], which avoids:
                                  1: buildings
                                  4: pedestrians
                                  5: poles
                                 10: vehicles
                                 11: walls
        @param horizon: the maximum distance from the camera that will checked
                        for collision (in km). This saves lots of computation
                        by avoiding checking for collision with the sky.
                        Defaults to 0.2 km (200 m).
        @param update_rate: in Hz, the frequency at which we update
        @param debug: set to True to enable debug logging.
        '''
        # Since we receive data from semantic segmentation and depth cameras
        # asynchronously, we need to create dictionaries to store the images
        # (indexed by frame)
        self.depth_data = dict()
        self.semantic_data = dict()

        # Initialize a place to store the trajectory
        self.trajectory = np.array([])

        # Save the transform from the vehicle to the camera
        self.camera_transform = camera_transform
        # Save the hazard labels
        self.hazard_labels = hazard_labels
        # Save the horizon
        self.max_depth = horizon
        # Save debug status
        self.debug = debug

        # Save information for timing updates
        self.last_update = time.time()
        assert(update_rate > 0)
        self.update_period = 1/update_rate

        # Instantiate a RefineNet instance
        print("before refine net init")
        self.refNet = RefineNet()
        print("after init")

    def log(self, message, emergency=False):
        '''Logs a message to the console if debug logging is enabled.

        @param message: the string message to print
        @param emergency: a boolean that should be set to true to ignore
                          current debug settings (usually we don't print unless
                          debug is enabled. Setting this to true prints anyway)
        '''
        # Do not print if debug is not enabled.
        if not (emergency or self.debug):
            return

        # Otherwise, add a timestamp to the message and print it
        flag = ""
        if emergency:
            flag = "[ALERT]"
        print(flag + "[BackseatDriver]" +
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

    def semantic_segmentation_callback(self, image_data):
        '''Receives semantic segmentation data asynchronously and saves it.

        @param image_data: a carla.Image object containing the RGB image.
                              We'll run it through RefineNet and save it here.
        TODO: Confirm label encoding with Lars
        '''
        # Convert the image to an RGB numpy array
        rgb_image = to_rgb_array(image_data)
        # Segment that image
        semantic_data = self.refNet.do_segmentation(rgb_image)

        # Save the depth data along with its frame
        self.semantic_data[image_data.frame] = semantic_data
        self.log("Received semantic data for frame: " +
                 str(image_data.frame))

    def update_planned_trajectory(self, trajectory):
        ''' "Files a flight plan" by telling the backseat driver what
        trajectory the car plans to follow in the near future.

        Following the common convention, the trajectory should be expressed
        in the ego vehicle frame.

        @param trajectory: an Nx4 numpy array, where each row is (t, x, y,
                           theta) denoting a time-indexed list of
                           trajectory waypoints, where
            - t is the time of the waypoint.
            - x, y, theta denotes the 2D pose of the vehicle at this waypoint,
              where (x, y, theta) = (0, 0, 0) is the current location of the
              vehicle (with the x-axis pointing along the current driving
              direction of the car).
        '''
        assert(trajectory.shape[1] == 4)
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
        # Skip if not enough time has elapsed since last update
        if time.time() - self.last_update < self.update_period:
            return

        self.last_update = time.time()
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
        distance_to_collision = np.inf
        if matching_frame_found:
            # Get the depth and semantic data from storage, and extract a
            # timestamp and other useful metadata for this frame
            depth_data = self.depth_data[frame_number]
            semantic_data = self.semantic_data[frame_number]
            # get the timestamp for these data and metadata
            timestamp = depth_data.timestamp

            # To save memory, now clear both storage dictionaries of any
            # frames less than the current frame number
            frames_to_delete = [key for key in self.depth_data.keys()
                                if key < frame_number]
            for frame in frames_to_delete:
                self.depth_data.pop(frame, None)

            frames_to_delete = [key for key in self.semantic_data.keys()
                                if key < frame_number]
            for frame in frames_to_delete:
                self.depth_data.pop(frame, None)

            # At this point, we have the depth and semantic data that we
            # want to fuse into a segmented point cloud.

            #   1) The semantic data is in an RGB array
            #      containing the label in the red channel (for interface with
            #      the depth_to_local_point_cloud function)

            #   2) Create a point cloud that contains only points
            #      that we labelled as hazards
            self.log("Making point cloud for frame " + str(frame_number))
            # Consider the following as hazards:
            #   1: buildings
            #   4: pedestrians
            #   5: poles
            #  10: vehicles
            #  11: walls
            point_cloud = depth_to_local_point_cloud(
                depth_data,
                semantic_data,
                max_depth=self.max_depth,
                hazard_labels=self.hazard_labels
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

            # Create the sub-trajectory to pass to the collision checker
            # Currently, the collision checker only considers x, y, and theta
            sub_trajectory = self.trajectory[start_index:, 1:]

            # Call the collision checker on the sub_trajectory
            distance_to_collision = get_collision(point_cloud.array,
                                                  sub_trajectory)

        if distance_to_collision != np.inf:
            self.log(("WARNING: collision predicted! Distance remaining (m): "
                      + str(distance_to_collision)), emergency=True)
        else:
            self.log("No collision predicted.")

        return distance_to_collision
