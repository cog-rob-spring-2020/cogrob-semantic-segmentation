import glob
import os
import sys

import numpy as np

from backseat_driver import BackseatDriver

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import argparse
import logging
import random


def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    args = argparser.parse_args()

    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    # ID for world.on_tick callback. Defined here so that it will be defined
    # during the finally block below if something goes wrong before we define
    # it in the try block.
    bsd_id = 0
    try:

        world = client.get_world()
        ego_vehicle = None
        ego_cam = None
        ego_depth = None
        ego_sem = None
        ego_col = None
        ego_lane = None
        ego_obs = None
        ego_gnss = None
        ego_imu = None

        # Define the transform for the RGBd camera
        cam_location = carla.Location(2, 0, 1)
        cam_rotation = carla.Rotation(-10, 180, 0)
        cam_transform = carla.Transform(cam_location, cam_rotation)

        # Create the BackseatDriver
        # No need to specify hazard labels; the defaults avoid cars,
        # pedestrians, buildings, walls, and poles
        my_backseat_driver = BackseatDriver(cam_transform, debug=True)

        # Register a simple "drive straight for a bit" trajectory.
        # Backseat driver will ignore waypoints with a time in the past, so
        # to force it to consider all waypoints, just add very large times
        trajectory = np.array([
            # t x y theta
            [np.inf, 0, 0, 0],
            [np.inf, 5, 0, 0],
            [np.inf, 10, 0, 0],
            [np.inf, 15, 0, 0],
            [np.inf, 20, 0, 0],
            [np.inf, 25, 0, 0]
        ])
        my_backseat_driver.update_planned_trajectory(trajectory)

        # --------------
        # Start recording
        # --------------
        """
        client.start_recorder('~/tutorial/recorder/recording01.log')
        """

        # --------------
        # Spawn ego vehicle
        # --------------
        ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
        ego_bp.set_attribute('role_name', 'ego')
        print('\nEgo role_name is set')
        ego_color = random.choice(
            ego_bp.get_attribute('color').recommended_values)
        ego_bp.set_attribute('color', ego_color)
        print('\nEgo color is set')

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if 0 < number_of_spawn_points:
            random.shuffle(spawn_points)
            ego_transform = spawn_points[0]
            ego_vehicle = world.spawn_actor(ego_bp, ego_transform)
            print('\nEgo is spawned')
        else:
            logging.warning('Could not found any spawn points')

        # --------------
        # Add a RGB camera sensor to ego vehicle.
        # --------------
        cam_bp = None
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x", str(1920))
        cam_bp.set_attribute("image_size_y", str(1080))
        cam_bp.set_attribute("fov", str(105))
        ego_cam = world.spawn_actor(
            cam_bp, cam_transform, attach_to=ego_vehicle,
            attachment_type=carla.AttachmentType.SpringArm)

        # Wire up the RGB camera to the backseat_driver callback
        ego_cam.listen(my_backseat_driver.semantic_segmentation_callback)
        print("rgb camera up")

        # --------------
        # Add a depth camera sensor to ego vehicle.
        # --------------
        depth_bp = None
        depth_bp = world.get_blueprint_library().find('sensor.camera.depth')
        depth_bp.set_attribute("image_size_x", str(1920))
        depth_bp.set_attribute("image_size_y", str(1080))
        depth_bp.set_attribute("fov", str(105))
        depth_location = carla.Location(2, 0, 1)
        depth_rotation = carla.Rotation(-10, 180, 0)
        depth_transform = carla.Transform(depth_location, depth_rotation)
        ego_depth = world.spawn_actor(
            depth_bp, depth_transform, attach_to=ego_vehicle,
            attachment_type=carla.AttachmentType.SpringArm)

        # Wire up the depth camera to the backseat_driver callback
        ego_depth.listen(my_backseat_driver.depth_callback)
        print("depth camera up")

        # --------------
        # Enable autopilot for ego vehicle
        # --------------
        ego_vehicle.set_autopilot(True)

        # --------------
        # Add the backseat driver to be called on every simulation tick
        # --------------
        bsd_id = world.on_tick(my_backseat_driver.get_safety_estimate)

        # --------------
        # Game loop. Prevents the script from finishing.
        # --------------
        while True:
            world_snapshot = world.wait_for_tick()

    finally:
        # --------------
        # Stop recording and destroy actors
        # --------------
        print("Cleaning up...")
        world.remove_on_tick(bsd_id)
        client.stop_recorder()
        if ego_vehicle is not None:
            if ego_cam is not None:
                ego_cam.stop()
                ego_cam.destroy()
            if ego_col is not None:
                ego_col.stop()
                ego_col.destroy()
            if ego_lane is not None:
                ego_lane.stop()
                ego_lane.destroy()
            if ego_obs is not None:
                ego_obs.stop()
                ego_obs.destroy()
            if ego_gnss is not None:
                ego_gnss.stop()
                ego_gnss.destroy()
            if ego_imu is not None:
                ego_imu.stop()
                ego_imu.destroy()
            if ego_sem is not None:
                ego_sem.stop()
                ego_sem.destroy()
            if ego_depth is not None:
                ego_depth.stop()
                ego_depth.destroy()
            ego_vehicle.destroy()


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\nDone with tutorial_ego.')
