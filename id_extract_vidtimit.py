import cv2
import face_recognition
import os
import torch
from model.base_model import get_model
import numpy as np
import pdb

def face_distance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.
    :param face_encodings: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    if len(face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encodings - face_to_compare, axis=1)

@torch.no_grad()
def inference(net, img):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    net.eval()
    img = img.cuda()
    net = net.cuda()
    feat = net(img).cpu().numpy()
    # print(feat)
    return feat

name = 'r50'
weight = '/nas/home/hliu/insightface/model_zoo/arcface_torch/ms1mv3_arcface_r50_fp16/backbone.pth'
arcface_net = get_model(name, fp16=False)
arcface_net.load_state_dict(torch.load(weight))
arcface_net = arcface_net.cuda()

# f = open('/nas/home/hliu/Datasets/FakeAVCeleb_v1.2/anomaly_frames_fake.txt', 'a')
f = open('/nas/home/hliu/Datasets/VidTIMIT/anomaly_frames.txt', 'a')

# root = "/nas/home/hliu/Datasets/FF++/Origin/"
# root = "/nas/home/hliu/Datasets/FakeAVCeleb_v1.2/FakeVideo-RealAudio/"
root = "/nas/home/dsalvi/VidTIMIT/"
# out = root.replace(root.split('/')[-2], root.split('/')[-2]+'_faceid128')
# out_arc = root.replace(root.split('/')[-2], root.split('/')[-2]+'_arcid512')
video_list = []

for s_dir in os.listdir(root):
    sub_dir =  os.path.join(root, s_dir)
    if os.path.isdir(sub_dir):
        video_list.append(sub_dir)

    # s_dir = os.path.join(root, qlt)
    # out_dir = os.path.join(out, qlt)
    # # arc_out_dir = os.path.join(out_arc, qlt)
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)
    # if not os.path.exists(arc_out_dir):
        # os.makedirs(arc_out_dir)
# import pdb
# pdb.set_trace
for video in video_list:
    video = os.path.join(video, 'video')
    out_arc = video.replace(video.split('/')[-3], video.split('/')[-3]+'_arcid512')
    out_arc = out_arc.replace('dsalvi', 'hliu/Datasets')
    for vid in os.listdir(video):
        img_path = os.path.join(video, vid)
        if not os.path.exists(out_arc):
            os.makedirs(out_arc)
        img_list = os.listdir(img_path)
        img_list.sort()
        img_length = len(img_list)
        arc_outname = os.path.join(out_arc, vid+'.npy')
        # arc_outname = os.path.join(arc_out_dir, vid+'.npy')
        print(img_path+' '+str(img_length))
        # face_id_flag = 0
        arc_id_flag = 0
        for i in range(img_length):
            img = os.path.join(img_path, img_list[i])
            # if face_id_flag == 0:
            #     try:
            #         face_id = face_recognition.load_image_file(img)
            #         face_id = face_recognition.face_encodings(face_id)[0]
            #         face_id = np.expand_dims(face_id, 0)
            #         face_id_flag = 1
            #     except:
            #         print('face id '+img+'\n')
            #         f.write('face id '+img+'\n')
            #         face_id_flag = 0
            # else:
            #     try:
            #         cur_face_id = face_recognition.load_image_file(img)
            #         cur_face_id = face_recognition.face_encodings(cur_face_id)[0]
            #         cur_face_id = np.expand_dims(cur_face_id, 0)
            #         face_id = np.concatenate((face_id, cur_face_id))
            #     except:
            #         print('face id '+img+'\n')
            #         f.write('face id '+img+'\n')

            if arc_id_flag == 0:
                try:
                    arc_id =  inference(arcface_net, img)
                    arc_id_flag = 1
                except:
                    print('arc id '+img+'\n')
                    f.write('arc id '+img+'\n')
                    arc_id_flag = 0
            else:
                try:
                    cur_arc_id = inference(arcface_net, img)
                    arc_id = np.concatenate((arc_id, cur_arc_id))
                except:
                    print('arc id '+img+'\n')
                    f.write('arc id '+img+'\n')
            # print(img)
            # import pdb
        # pdb.set_trace()
        # np.save(outname, face_id)
        # print(outname+' is saved!')
        np.save(arc_outname, arc_id)
        print(arc_outname+' is saved!')
f.close()
