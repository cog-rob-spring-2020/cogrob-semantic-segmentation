import numpy as np
from collision.utils import *


def pi2pi(theta):
    return (theta+np.pi) % np.pi-np.pi


def cal_curvature(p1, p2):
    '''
    calculate curvature, p1 current waypoint, p2 next waypoint
    '''
    x1, y1, theta1 = p1[:3]
    x2, y2, theta2 = p2[:3]

    theta0 = np.arctan2(y2-y1, x2-x1)
    gamma1 = theta1 - theta0
    gamma2 = theta0 - theta2

    d = np.sqrt((y2-y1)**2+(x2-x1)**2)

    rho = np.sin(gamma1)/d*2
    return rho


def interpolate_waypoints(p1, p2):
    '''
    calculate curvature, p1 current waypoint, p2 next waypoint
    '''
    new_waypoints = []

    x1, y1, theta1 = p1[:3]
    x2, y2, theta2 = p2[:3]

    theta0 = np.arctan2(y2-y1, x2-x1)
    gamma1 = theta1 - theta0
    gamma2 = theta0 - theta2

    if gamma1 == gamma2:
        return [[x1, y1, theta1], [x2, y2, theta2]]

    d = np.sqrt((y2-y1)**2+(x2-x1)**2)

    mid_point = np.array([(x1+x2)/2., (y1+y2)/2.])

    vec = np.cross([x2-x1, y2-y1, 0], [0, 0, 1.])[:2]
    vec = vec/np.linalg.norm(vec)

    beta = (gamma1+gamma2)/4.

    x_new, y_new = mid_point - np.tan(beta)*d/2 * vec

    theta_new = theta0 + 2*beta - gamma1

    return [[x1, y1, theta1], [x_new, y_new, theta_new], [x2, y2, theta2]]


def process_trajectory(trajectory):
    new_traj = [trajectory[0][:3]]
    for p1, p2 in zip(trajectory, trajectory[1:]):
        new_traj += interpolate_waypoints(p1, p2)[1:]

    new_traj2 = []
    for p1, p2 in zip(new_traj, new_traj[1:]):
        p1_new = p1 + [cal_curvature(p1, p2)]
        new_traj2.append(p1_new)
    new_traj2.append(p2+[p1_new[-1]])

    return new_traj2


def waypoints_distance(p1, p2):
    x1, y1, theta1, rho = p1
    x2, y2, theta2, _ = p2

    d = np.sqrt((y2-y1)**2+(x2-x1)**2)

    if abs(rho) < 0.001:
        length = d/np.sqrt(4.-d**2*rho**2)*2  # 1'st order approximation
    else:
        length = 2*np.arcsin(d*rho/2)/rho

    return length


def transformation(p, point_cloud):
    x,y,theta = p[:3]
    R_cw = np.array([[np.sin(theta), -np.cos(theta)],[np.cos(theta), np.sin(theta)]]).T
    x_cw = np.array([x,y])
    point_cloud_c = np.dot(R_cw.T, point_cloud.T-x_cw.reshape((2,1)))
    return point_cloud_c.T



def get_collision(point_cloud, trajectory, margin = 1.):
    '''
    Interface function of collision detection
    Input:
        point_cloud: Nx3 ndarray [[x,y,z], ...] 
        trajectory: list of tuple [(x,y,theta), ...]
            x: x component of car position in world frame
            y: y component of car position in world frame
            theta: orientation of car in world frame
                     when theta=0 car moving along x-axis (world frame), theta increase counterclockwise
    Output:
        maximum distance the vehicle can go, inf if no collision has been detected
    '''

    point_cloud = np.array(point_cloud)[:,:2]

    trajectory = process_trajectory(trajectory)

    total_traveled = 0.
    for p1,p2 in zip(trajectory,trajectory[1:]):
        point_cloud_p = transformation(p1,point_cloud)
        d = get_distance(p1[3], point_cloud_p, margin)

        wd = waypoints_distance(p1,p2)

        if d<wd:
            return total_traveled+d
        else:
            total_traveled += wd

    return np.inf
    pass


def get_collision2(depth_image, seg_image, trajectory):
    '''
    Interface function of collision detection
    Input:
        depth_image: 2D ndarray depth_image[height, width] = distance of pixel (height, width) to camera
        seg_image: 2D ndarray seg_image[height, width] = semantic class index of pixel (height, width)
        trajectory: (TBD) list of tuple [(x,y,theta, rho), ...]
            x: x component of car position in world frame
            y: y component of car position in world frame
            theta: orientation of car in world frame
                     theta=0 is x-axis (world frame), theta increase counterclockwise
    Output:
        collision_image: 2D ndarray, collision distance represent in image
        collision_image[height, width] = distance to collision on this pixel (inf if the car will not collide to this pixel)
    '''

    # This function is currently not working. Use get_collision instead

    instance_list = segment(depth_image, seg_image)
    objects = get_bounding_bos(instance_list)
    return get_distance(objects)

    pass

#The following codes is only used for plotting and debugging
import matplotlib.pyplot as plt

def plot_pose(p):

    x,y,theta = p[:3]

    plt.plot(x,y,'k.',markersize = 10)

    l = 0.3

    x1 = l*np.cos(theta)+x
    y1 = l*np.sin(theta)+y

    plt.plot([x,x1],[y,y1],'-b')

def plot_curve(p1, p2):
    x1,y1,theta1, rho = p1
    x2,y2,theta2, _ = p2

    if abs(rho)<0.001:
        plt.plot([x1,x2],[y1,y2],'-r')
    else:
        r = 1./rho
        xc,yc = np.cross([np.cos(theta1),np.sin(theta1),0],[0,0,1.])[:2]*r + np.array([x1,y1])
        # print xc,yc
        # plt.plot(xc,yc,'o')
        # d = np.sqrt((y2-y1)**2+(x2-x1)**2)
        # dtheta = 2*np.asin(d*rho/2)

        alpha1 = np.arctan2(y1-yc, x1-xc)
        alpha2 = np.arctan2(y2-yc, x2-xc)
        # print alpha1,alpha2
        if r>0:
            if alpha2>alpha1:
                alpha2 -= 2*np.pi

            dalpha = np.linspace (0., alpha2-alpha1, num=10)
        else:
            if alpha1>alpha2:
                alpha1-=2*np.pi
            dalpha = np.linspace (0., alpha2-alpha1, num=10)

        plt.plot(np.cos(alpha1+dalpha)*abs(r)+xc, np.sin(alpha1+dalpha)*abs(r)+yc, '-r')



def plot_trajectory(traj):

    for p1,p2 in zip(traj, traj[1:]):
        plot_curve(p1,p2)
    for p in traj:
        # pass
        plot_pose(p)
    pass

def gen_obstacle(p,num=40):
    x,y,r = p
    theta = np.random.rand(num)*np.pi*2
    xx = r*np.cos(theta)+x
    yy = r*np.sin(theta)+y
    point_cloud = np.array([xx,yy]).T
    return point_cloud

def gen_multiple_obstacle(p_list):
    point_clouds = []
    for p in p_list:
        point_clouds.append(gen_obstacle(p))
    point_clouds = np.concatenate(point_clouds)
    return point_clouds

def plot_pointcloud(pc):
    plt.plot(pc[:,0],pc[:,1],'.k')

def test():
    
    plt.figure()

    traj = [[0,0,np.pi/4],[1,1,np.pi/4], [2,2,np.pi/4], [3,4,np.pi/2], [5,4,np.pi/2], [5,7,np.pi/2]]
    trajectory = process_trajectory(traj)
    # plot_pose([0,0,1.])

    # plot_curve([0,0,np.pi/2, 1],[1,1,0.,1])

    plot_trajectory(trajectory)

    pc = gen_multiple_obstacle([[2,3,0.5],[3,5,0.5]])
    plot_pointcloud(pc)
    print(get_collision(pc,trajectory,0.5))
    plt.show()

if __name__ == '__main__':
    # print(get_rot_center(3.14/6))
    # obstacle = [[0., 1.], [0.8, 0.8]]
    # print(get_distance(-1.14/6, obstacle))
    test()
