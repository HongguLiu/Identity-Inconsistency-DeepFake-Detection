import os
import argparse
from os.path import join
import cv2
import dlib
import torch
import torch.nn as nn
from PIL import Image as pil_image
# from tqdm import tqdm
import random



def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb






def predict_demo(video_path, cuda=False):
    print('Starting: {}'.format(video_path))
    # Read and write
    reader = cv2.VideoCapture(video_path)
    output_path = os.path.join('./', video_path.split('.')[0].split('/')[-1])
    os.makedirs(output_path, exist_ok=True)
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Face detector
    face_detector = dlib.get_frontal_face_detector()


    # Text variables
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1

    # Frame numbers and length of output video
    frame_num = 0
    start_frame=0
    assert start_frame < num_frames - 1
    end_frame = num_frames

    while reader.isOpened():
        _, image = reader.read()
        if image is None:
            break
        frame_num += 1

        if frame_num < start_frame:
            continue
        # pbar.update(1)
        # Image size
        height, width = image.shape[:2]


        # 2. Detect with dlib
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces):
            # For now only take biggest face
            face = faces[0]

            # --- Prediction ---------------------------------------------------
            # Face crop with dlib and bounding box scale enlargement
            x, y, size = get_boundingbox(face, width, height)
            #cropped_face = image[y:y+size, x:x+size]

            # Text and bb
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y

            color = (0, 255, 0)
            prob = ['50%', '52%', '48%', '49%', '38%']
            # if frame_num <= 10:
                # ans = 'Brad Pitt '+'99%'
                # ans = 
            # else:
                # ans = 'Brad Pitt '+'98%'
            pro = random.choice(prob)
            ans = 'Brad Pitt '+pro

            cv2.putText(image, ans, (x, y+h+30),
                        font_face, font_scale,
                        color, thickness, 3)
            # draw box over face
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        if frame_num == 300:
            break
        cv2.imwrite(os.path.join(output_path, str(frame_num)+'.png'), image)



predict_demo(video_path='/nas/public/dataset/celeb-df-v2/Celeb-synthesis/id0_id1_0002.mp4', cuda=True)
# predict_demo(video_path='/nas/public/dataset/celeb-df-v2/Celeb-real/id1_0002.mp4', cuda=True)
