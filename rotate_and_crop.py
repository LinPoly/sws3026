import os
import cv2 as cv
import dlib
import numpy as np
from skimage.feature import hog
import sklearn
from matplotlib import pyplot as plt

# T1  start _______________________________________________________________________________
# Read in Dataset

# change the dataset path here according to your folder structure
dataset_path = "Dataset_1"

X = []
y = []

# naming of subjects
subject_prefix = "s"
subject_range = (1, 51)

for subject_index in range(*subject_range):

    subject_name = f"{subject_prefix}{subject_index // 10}{subject_index % 10}"

    y.append(subject_name)
    subject_images_dir = f"{dataset_path}/{subject_name}"

    temp_x_list = []
    for img_name in os.listdir(subject_images_dir):
        # write code to read each 'img'
        if not img_name.endswith('.jpg'):
            continue
        img_path = os.path.join(subject_images_dir, img_name)
        img = dlib.load_rgb_image(img_path)
        # add the img to temp_x_list
        temp_x_list.append(img)
    # add the temp_x_list to X
    X.append(temp_x_list)
# T1 end ____________________________________________________________________________________

# T2 start __________________________________________________________________________________
# Preprocessing

X_processed = []
detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
nomask_data_dir = "output/dataset_nomask"

for i, x_list in enumerate(X):
    temp_X_processed = []
    nomask_path = os.path.join(nomask_data_dir, f"s{i}")
    os.makedirs(nomask_path)
    for j, x in enumerate(x_list):
        # write the code to detect face in the image (x) using dlib facedetection library
        detector = dlib.get_frontal_face_detector()
        dets = detector(x, 2)
        det = dets[0]
        x = dlib.as_grayscale(x)
        shape = predictor(x, det)
        # write the code to crop the image (x) to keep only the face, resize the cropped image to 150x150
        temp_points = np.array(shape.parts())
        points: np.ndarray = np.ndarray(shape=(0, 2), dtype=np.uint8)
        for p in temp_points:
            points = np.vstack([points, [p.x, p.y]])
        x_axis, y_axis = points[:, 0], points[:, 1]

        # rotate face by feature points
        # the feature point chosen are No.8 (the mid point of jaw line) and No.27 (the top point of nose)
        center = (points[27] + points[8]) / 2
        center = tuple(center)
        orient_vec: np.ndarray = points[27] - points[8]
        angle = np.sign(orient_vec[0]) * np.arccos(-orient_vec[1] /
                                                   np.linalg.norm(orient_vec))
        # perform transform
        rotation_matrix = cv.getRotationMatrix2D(
            center=center, angle=angle * 180 / np.pi, scale=1)
        x = cv.warpAffine(
            src=x, M=rotation_matrix, dsize=(x.shape[1], x.shape[0]))
        points = np.vstack([points.T, [1] * points.shape[0]])
        points = (rotation_matrix @ points).T

        # Cut the image to square to avoid distortion
        x_axis, y_axis = points[:, 0], points[:, 1]
        left, right = np.min(x_axis), np.max(x_axis)
        top, bottom = np.min(y_axis), np.max(y_axis)
        w, h = right - left, bottom - top
        if w > h:
            top -= (w - h) / 2
            bottom += (w - h) / 2
        else:
            left -= (h - w) / 2
            right += (h - w) / 2
        margin_ratio = 0.05  # keep margin or not
        resized_x = x[int(top - margin_ratio * w):int(bottom + margin_ratio * w),
                      int(left - margin_ratio * w):int(right + margin_ratio * w)]
        resized_x = dlib.resize_image(resized_x, 150, 150)
        cv.imwrite(os.path.join(nomask_path, f"{j}.jpg"), resized_x)

        temp_X_processed.append(resized_x)
    X_processed.append(temp_X_processed)

# T2 end ____________________________________________________________________________________


# T3 start __________________________________________________________________________________
# Create masked face dataset
X_masked = []
shp_det = dlib.rectangle(0, 0, 149, 149)
mask_data_path = "output/dataset_mask"

for i, x_list in enumerate(X_processed):
    temp_X_masked = []
    mask_path = os.path.join(mask_data_path, f"s{i}")
    os.makedirs(mask_path)
    for j, x in enumerate(x_list):
        # write the code to detect face in the image (x) using dlib facedetection library
        local_shape = predictor(x, shp_det)
        key_points = local_shape.parts()
        temp_points = key_points[1:16]
        temp_points.extend([key_points[35], key_points[28], key_points[31]])
        mask_points = np.ndarray(shape=(0, 2), dtype=np.uint8)
        for p in temp_points:
            mask_points = np.vstack([mask_points, [p.x, p.y]])
        # write the code to add synthetic mask as shown in the project problem description
        cv.fillPoly(x, [mask_points], color=(255, 255, 255))
        # append the converted image into temp_X_masked
        cv.imwrite(os.path.join(mask_path, f"{j}.jpg"), x)
        temp_X_masked.append(x)
    # append temp_X_masked into  X_masked
    X_masked.append(temp_X_masked)
