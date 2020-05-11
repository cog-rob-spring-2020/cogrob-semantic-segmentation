import numpy as np

FRONT_BACK_DISTANCE = 2.
max_detect_range = np.inf
# def get_rot_center(alpha):
#     '''
#     Calculate the rotation center of the vehicle
#     Input: alpha is the servo angle (angle between front wheel and front
#     (clockwise is positive, 0 is straight forward))
#     Output: Rotation Center in Camera Frame
#     '''
#     if abs(alpha) < 0.01:
#         return None

#     # car_uaw = 0.04  # y component of the Front wheel in camera Frame
#     # car_aw = -0.28  # y component of the rear wheel in camera Frame
#     # fw = car_uaw
#     # bw = car_aw

#     x = FRONT_BACK_DISTANCE/np.sin(alpha)
#     y = -FRONT_BACK_DISTANCE
#     return (x, y)

def pi2pi(theta):
    return (theta+np.pi) % (2*np.pi)-np.pi

def get_rot_center(rho, x=0., y=0., theta = np.pi/2):
    '''
    Calculate the rotation center of the vehicle
    Input: rho is the curvature at current state, positive is right turning, negitive is left turning
    Output: Rotation Center in Camera Frame
    '''
    if abs(rho) < 0.01:
        return None

    car_aw = 0.
    bw = car_aw  # y component of the rear wheel in camera Frame

    x = 1./rho
    y = bw
    return (x, y)


def get_distance(rho, obstacle, margin=1., noise_level=0):
    '''
    Calculate the minimum distance that the vehicle could move without collision
    Input:  rho: curvature at current state, positive is right turning, negitive is left turning
            obstacle: lidar points list in Polar coordinates (camera frame)
                [[beta1, r1], [beta2, r2]]
                beta is the angle, r is the distance
            margin: safety margin of the object. The vehicle will be considered as collision if it intersect the margin region (a disk with margin as radius) of any lidar point
            noise_level: number of lidar point that will be considered as noise,  set to 0 as default
    Output: Minimum distance that the vehicle could move without collision
    '''

    obstacle = np.array(obstacle)
    # beta = obstacle[:, 0]
    # d = obstacle[:, 1]
    # print obstacle

    # print min(np.sqrt(obstacle[:,1]**2+obstacle[:,0]**2))

    if min(np.sqrt(obstacle[:,1]**2+obstacle[:,0]**2))<=margin:
        return 0.

    obstacle = obstacle[np.where(obstacle[:,1]>=0)]
    # print obstacle

    xc = obstacle[:,0]
    yc = obstacle[:,1]

    center = get_rot_center(rho)
    # print center
    if not center:
        # The vehicle is moving forward

        # x = d*np.sin(beta)
        # y = d*np.cos(beta)

        x = xc
        y = yc

        crossing = np.where(np.abs(x) > margin, 0, 1)
        collide_indx = np.where(crossing == 1)
        if len(collide_indx[0]) == 0:
            return max_detect_range
        x = np.where(np.abs(x) <= margin, x, margin)
        distance = y-np.sqrt(margin*margin-x*x)
        distance_list = sorted(distance[collide_indx])
        # return np.min(distance[collide_indx])
        if len(distance_list) <= noise_level:
            return max_detect_range
        return distance_list[noise_level]

    x0, y0 = center
    r0 = np.sqrt(x0**2+y0**2)

    # x = -x0+d*np.sin(beta)
    # y = -y0+d*np.cos(beta)

    x = -x0 + xc
    y = -y0 + yc

    # max_go_range = (np.pi-np.abs(np.arctan(y0/x0)))*r0
    max_go_range = np.inf
    # print 'max_go_range',max_go_range,'center',center

    # direction=np.sign(rho)
    if rho > 0:
        # The vehicle is turning right

        rho0 = np.sqrt(x*x+y*y)
        theta_obstacle = np.arctan(y/x)
        theta_robot = np.arctan(y0/x0)
        theta_obstacle = np.where(
            theta_obstacle < 0, theta_obstacle+np.pi, theta_obstacle)
        if theta_robot < 0:
            theta_robot += np.pi

        cos_dtheta = (rho0*rho0+r0**2-margin**2)/2/rho0/r0

        crossing = np.where(np.abs(cos_dtheta) <= 1, 1, 0)
        collide_indx = np.where(crossing == 1)

        if len(collide_indx[0]) == 0:
            return min(max_detect_range, max_go_range)
        cos_dtheta = np.where(np.abs(cos_dtheta) <= 1, cos_dtheta, 1)
        dtheta = np.arccos(cos_dtheta)

        theta_crossing = theta_robot-dtheta
        # print theta_crossing,'tc'
        # print theta_obstacle,'tb'

        distance = (theta_crossing-theta_obstacle)*r0
        distance[np.where(distance<0)] += r0*np.pi
    else:
        # The vehicle is turning left

        rho0 = np.sqrt(x*x+y*y)
        theta_obstacle = np.arctan(y/x)
        theta_robot = np.arctan(y0/x0)
        theta_obstacle = np.where(
            theta_obstacle < 0, theta_obstacle+np.pi, theta_obstacle)
        if theta_robot < 0:
            theta_robot += np.pi

        cos_dtheta = (rho0*rho0+r0**2-margin**2)/2/rho0/r0

        crossing = np.where(np.abs(cos_dtheta) <= 1, 1, 0)
        collide_indx = np.where(crossing == 1)

        if len(collide_indx[0]) == 0:
            return min(max_detect_range, max_go_range)
        cos_dtheta = np.where(np.abs(cos_dtheta) <= 1, cos_dtheta, 1)
        dtheta = np.arccos(cos_dtheta)

        theta_crossing = theta_robot+dtheta

        distance = -(theta_crossing-theta_obstacle)*r0
        distance[np.where(distance<0)] += r0*np.pi

    # distance to all lidar points
    # print obstacle[collide_indx]
    distance_list = sorted(distance[collide_indx])
    # return the minimum
    return min(distance_list[min(noise_level, len(distance_list)-1)], max_go_range)


def get_distance_3D_baseline(rho, obstacle, margin=0.5, noise_level=0):
    '''
    Calculate the minimum distance that the vehicle could move without
    causing a collision with specified obstacles. A very simple function
    that does not account for vehicle turning. To be used until the proper
    collision checking code is complete.

    The vehicle is assumed to be moving forward along the local x axis.

    Input:  rho: curvature at current state, positive is right turning,
                 negative is left turning.
            obstacle: a point cloud representing obstacles, with points encoded
                      as 3D coordinates in a N_points x 3 np array
                          [[x1, y1, z1], [x2, y2, z2], ...]
            margin: safety margin of the object. The vehicle will be considered
                    as in collision if it intersects the margin region (a disk
                    with margin as radius) of any point
            noise_level: number of lidar point that will be considered as
                         noise,  set to 0 as default
    Output: Minimum distance that the vehicle could move without collision
    '''

    # Check for collision along the
    # local x axis (pointing forward)
    x = obstacle[:, 0]
    y = obstacle[:, 1]
    z = obstacle[:, 2]
    # Extract the indices of points where collision might occur.
    crossing = np.where((np.abs(y) > margin) & (np.abs(z) > margin), 0, 1)
    collide_indx = np.where(crossing == 1)
    # If no collisions, return infinity
    if len(collide_indx[0]) == 0:
        return np.inf
    # Otherwise, find the minimum x at which a collision occurs, and return
    # that as our "distance to collision"
    return np.min(x[collide_indx])


def segment(depth_image, seg_image):
    pass


def get_bounding_bos(instance_list):
    pass


if __name__ == '__main__':
    # print(get_rot_center(3.14/6))
    # obstacle = [[0., -1.], [0.8, 0.8]]
    obstacle = [[ 2.32682892e-16,  1.90000000e+00],
    [ 3.21866275e-02,  2.69870292e+00],
                [-3.21866275e-02,  2.69870292e+00],
                [-9.57262657e-02,  2.68837673e+00],
                [-1.56786644e-01,  2.66799178e+00],
                [-2.13786330e-01,  2.63807603e+00],
                [ 2.32682892e-16 , 1.90000000e+00],
                [-2.65249063e-01,  2.59940430e+00]]
    # obstacle = [[0,2]]
    print(get_distance(0.1, obstacle))
