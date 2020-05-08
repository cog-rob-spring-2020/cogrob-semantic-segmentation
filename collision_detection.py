import numpy as np
from collision.utils import *

def get_collision(depth_image, seg_image, trajectory):
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
            rho: Curvature of at (x, y), assume the curvature is constant until next trajectory point (during movement from (x_t, y_t) to (x_t+1, y_t+1))
                The curvature is positive when turning right, negative when turning left (TBD) 
    Output:
        collision_image: 2D ndarray, collision distance represent in image
        collision_image[height, width] = distance to collision on this pixel (inf if the car will not collide to this pixel)
    '''

    #This function is currently not working. I will finish it by Saturday -shangjie

    instance_list = segment(depth_image,seg_image)
    objects = get_bounding_bos(instance_list)
    return get_distance(objects)

    
    pass

if __name__ == '__main__':
    print(get_rot_center(3.14/6))
    obstacle = [[0., 1.], [0.8, 0.8]]
    print(get_distance(-1.14/6, obstacle))
