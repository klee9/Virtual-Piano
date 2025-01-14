import numpy as np
import json
import torch
import cv2
import os

from blazebase import resize_pad, denormalize_detections
from blazepalm import BlazePalm
from visualization import draw_detections, draw_roi
from blazehand_landmark import BlazeHandLandmark

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

back_detector = True

palm_detector = BlazePalm().to(gpu)
palm_detector.load_weights("/Users/klee9/Desktop/daiv/kirby/models/blazepalm.pth")
palm_detector.load_anchors("/Users/klee9/Desktop/daiv/kirby/models/anchors_palm.npy")
palm_detector.min_score_thresh = .75

hand_regressor = BlazeHandLandmark().to(gpu)
hand_regressor.load_weights("/Users/klee9/Desktop/daiv/kirby/models/blazehand_landmark.pth")

# train images
train_imgs = "/Users/klee9/Downloads/FreiHAND_pub_v2/training/rgb"

# save paths
train_abs_xyz = "/Users/klee9/Desktop/daiv/kirby/train_abs_xyz.json"
eval_abs_xyz = "/Users/klee9/Desktop/daiv/kirby/eval_abs_xyz.json"

train_normalized_path = "/Users/klee9/Desktop/daiv/kirby/train_normalized_xyz.json"
eval_normalized_path = "/Users/klee9/Desktop/daiv/kirby/eval_normalized_xyz.json"

train_save_path = "/Users/klee9/Desktop/daiv/kirby/train_coco_xyz.json"
eval_save_path = "/Users/klee9/Desktop/daiv/kirby/eval_coco_xyz.json"

# label paths
train_xyz = "/Users/klee9/Desktop/daiv/kirby/dataset/training_xyz.json"
eval_xyz = "/Users/klee9/Desktop/daiv/kirby/dataset/evaluation_xyz.json"
K_file_path = '/Users/klee9/Downloads/FreiHAND_pub_v2/training_K.json'

def absolutify(labels, save_path):
    # load json labels
    with open(labels, 'r') as file:
        xyz_data = json.load(file)

    with open(K_file_path, 'r') as file:
        K_data = json.load(file)

    # calculate absolute coordinates
    absolute_coordinates = []
    for i, frame_landmarks in enumerate(xyz_data):
        K = np.array(K_data[i])
        frame_abs_coords = []
        for landmark in frame_landmarks:
            x, y, z = landmark
            
            # 2D projection
            normalized_coords = np.array([x / z, y / z, 1.0])
            projected_coords = K @ normalized_coords 
            u, v = projected_coords[:2]
            
            frame_abs_coords.append([u, v, z])
        
        absolute_coordinates.append(frame_abs_coords)

    # save
    with open(save_path, 'w') as output_file:
        json.dump(absolute_coordinates, output_file)

    print(f"absolute coordinates saved to {save_path}")


def normalize(labels, save_path, n=224):
    # use absolute coordinates to calculate normalized coordinates
    with open(labels, 'r') as file:
        abs_coords = json.load(file)
    
    normalized_coords = []
    for frame_landmarks in abs_coords:
        frame_normalized_coords = []
        for landmark in frame_landmarks:
            x, y, z = landmark
            x, y, z = x/n, y/n, z/n
            frame_normalized_coords.append([x, y, z])

        normalized_coords.append(frame_normalized_coords)

    # save
    with open(save_path, 'w') as output_file:
        json.dump(normalized_coords, output_file)

    print(f"normalized coordinates saved to {save_path}")


# extract pos, scale values (for original cocofolder code)
def extract_pos_scale(img_folder):
    failed_imgs = []
    for img_path in sorted(os.listdir(img_folder)):
        full_img_path = os.path.join(img_folder, img_path)

        img = cv2.imread(full_img_path)

        img1, img2, scale, pad = resize_pad(img)
        normalized_palm_detections = palm_detector.predict_on_image(img1)
        palm_detections = denormalize_detections(normalized_palm_detections, scale, pad)
        xc, yc, scale, theta = palm_detector.detection2roi(palm_detections.cpu())

        # skip when detection fails
        if xc.numel() == 0 or yc.numel() == 0: 
            failed_imgs.append(img_path)
            continue

    print(f"Failed to detect palm in: {failed_imgs}")


def cvt2coco(input_path, output_path):
    with open(input_path, 'r') as file:
        data = json.load(file)

    # match cocofolder structure
    transformed_data = [ {"info": [{"keypoints": keypoints}]} for keypoints in data ]

    # save
    with open(output_path, 'w') as file:
        json.dump(transformed_data, file, indent=2)

    print(f"fully converted file saved to {output_path}")


absolutify(train_xyz, train_abs_xyz)
absolutify(eval_xyz, eval_abs_xyz)

normalize(train_abs_xyz, train_normalized_path)
normalize(eval_abs_xyz, eval_normalized_path)

cvt2coco(train_normalized_path, train_save_path)
cvt2coco(eval_normalized_path, eval_save_path)
