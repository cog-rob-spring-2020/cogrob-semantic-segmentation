import cv2
import os
import numpy as np
import csv


classes = {
    0: [0, 0, 0],         # None
    1: [0, 64, 192],      # Buildings
    2: [190, 153, 153],   # Fences
    3: [72, 0, 90],       # Other
    4: [128, 128, 192],     # Pedestrians
    5: [0, 0, 128],   # Poles
    6: [157, 234, 50],    # RoadLines
    7: [128, 64, 128],    # Roads
    8: [244, 35, 232],    # Sidewalks
    9: [107, 142, 35],    # Vegetation
    10: [128, 128, 128],      # Vehicles
    11: [0, 192, 192],  # Walls
    12: [220, 220, 0]     # TrafficSigns
}

rgb_image_folder = '/media/eric/Western/carla/run3/presentation/rgb'
ss_image_folder = '/media/eric/Western/carla/run3/presentation/ss'
video_name = 'video.avi'

reader = csv.reader(open('/media/eric/Western/carla/run3/distances.csv'))

distances_to_collision = {}
for row in reader:
    key = "00"+row[0]
    distances_to_collision[key] = row[1]

images = [img for img in os.listdir(rgb_image_folder) if img.endswith(".png")]
images = sorted(images)
frame = cv2.imread(os.path.join(rgb_image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 10, (width,height))

def overlay_danger(image_ss_fname, image_rgb_fname):
    alpha = .35
    image_ss = cv2.imread(image_ss_fname)
    image_rgb = cv2.imread(image_rgb_fname)

    car_color = np.array(classes[10])
    ped_color = np.array(classes[4])
    bld_color = np.array(classes[1])
    pol_color = np.array(classes[5])
    wal_color = np.array(classes[11])
    car_mask = cv2.inRange(image_ss, car_color, car_color)
    ped_mask = cv2.inRange(image_ss, ped_color, ped_color)
    bld_mask = cv2.inRange(image_ss, bld_color, bld_color)
    pol_mask = cv2.inRange(image_ss, pol_color, pol_color)
    wal_mask = cv2.inRange(image_ss, wal_color, wal_color)
    mask = cv2.bitwise_or(car_mask, ped_mask)
    mask = cv2.bitwise_or(mask, bld_mask)
    mask = cv2.bitwise_or(mask, pol_mask)
    mask = cv2.bitwise_or(mask, wal_mask)
    overlay = cv2.bitwise_and(image_ss, image_ss, mask = mask)

    return cv2.addWeighted(overlay, alpha, image_rgb, 1 - alpha, 0)

last_distance = np.inf
for image in images:
    rgb_image_fname = os.path.join(rgb_image_folder, image)
    ss_image_fname = os.path.join(ss_image_folder, image)
    frame = overlay_danger(ss_image_fname, rgb_image_fname)
    frame_number = image.split(".")[0]
    if frame_number in distances_to_collision:
        last_distance = distances_to_collision[frame_number]
    cv2.putText(frame, "DTC (m): {:.1f}".format(float(last_distance)), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    video.write(frame)

cv2.destroyAllWindows()
video.release()
