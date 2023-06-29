import cv2
import os
import torch
import numpy as np
import pdb
import matplotlib.pyplot as plt

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def face_distance(face_encodings, face_to_compare):
    # if len(face_encodings) == 0:
        # return np.empty((0))

    return np.linalg.norm(face_encodings - face_to_compare, axis=0)

cdf_real_root = "/nas/home/hliu/Datasets/FF++/Origin_arcid512/raw/"
cdf_fake_root = "/nas/home/hliu/Datasets/FF++/Deepfakes_arcid512/raw/"

real_list = os.listdir(cdf_real_root)
fake_list = os.listdir(cdf_fake_root)
real_list.sort()
fake_list.sort()

real_diff = []
fake_diff = []

# for vid in real_list:
#     vid_id = np.load(os.path.join(cdf_real_root, vid))
#     vid_score = AverageMeter()
#     vid_score.reset()
#     length = vid_id.shape[0]
#     # pdb.set_trace()
#     for i in range(length-1):
#         vid_score.update(val=face_distance(vid_id[i], vid_id[i+1]))
#     print(vid_score.avg)
#     real_diff.append(vid_score.avg)

# fake_list = fake_list[::5]

# for vid in fake_list:
#     vid_real = vid.split('_')[-1]
#     vid_id = np.load(os.path.join(cdf_fake_root, vid))
#     vid_score = AverageMeter()
#     vid_score.reset()
#     length = vid_id.shape[0]
#     for i in range(length-1):
#         vid_score.update(val=face_distance(vid_id[i], vid_id[i+1]))
#     # print(vid_score.avg)
#     fake_diff.append(vid_score.avg)

#     vid_id = np.load(os.path.join(cdf_real_root, vid_real))
#     vid_score = AverageMeter()
#     vid_score.reset()
#     length = vid_id.shape[0]
#     for i in range(length-1):
#         vid_score.update(val=face_distance(vid_id[i], vid_id[i+1]))
#     # print(vid_score.avg)
#     real_diff.append(vid_score.avg)

id_real=[]
id_fake=[]
vid = fake_list[1]
vid_real = vid.split('_')[0].split('/')[-1]+'.npy'
vid_id = np.load(os.path.join(cdf_fake_root, vid))
# vid_score = AverageMeter()
# vid_score.reset()
length = vid_id.shape[0]
for i in range(length-1):
    # vid_score.update(val=face_distance(vid_id[i], vid_id[i+1]))
    id_fake.append(face_distance(vid_id[i], vid_id[i+1]))
    # print(vid_score.avg)
# fake_diff.append(vid_score.avg)

vid_id = np.load(os.path.join(cdf_real_root, vid_real))
# vid_score = AverageMeter()
# vid_score.reset()
length = vid_id.shape[0]
for i in range(length-1):
    id_real.append(face_distance(vid_id[i], vid_id[i+1]))
    # vid_score.update(val=face_distance(vid_id[i], vid_id[i+1]))
    # print(vid_score.avg)
# real_diff.append(vid_score.avg)



# pdb.set_trace()

x = np.arange(0,length-1)
label=['Origin ID', 'DeepFake ID']
print(np.var(id_fake), '--------', np.var(id_real))
plt.plot(x, id_real, 'g', x, id_fake, 'r', x, np.array(id_fake)-np.array(id_real), 'y')
# plt.legend(loc="upper right")
plt.legend(label)
plt.savefig('./images/000_DF_ORG.png')

