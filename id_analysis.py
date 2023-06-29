import cv2
import os
import torch
import numpy as np
import pdb
import matplotlib.pyplot as plt
from numpy import polyfit, poly1d

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
cdf_fake_root = "/nas/home/hliu/Datasets/FF++/FaceSwap_arcid512/raw/"

real_list = os.listdir(cdf_real_root)
fake_list = os.listdir(cdf_fake_root)
real_list.sort()
fake_list.sort()

real_diff = []
fake_diff = []

real_var = []
fake_var = []

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

fake_list = fake_list[::10]

for vid in fake_list:
    fake_id = []
    vid_real = vid.split('_')[-1]
    vid_id = np.load(os.path.join(cdf_fake_root, vid))
    vid_score = AverageMeter()
    vid_score.reset()
    length = vid_id.shape[0]
    for i in range(length-1):
        id_dis = face_distance(vid_id[i], vid_id[i+1])
        fake_id.append(id_dis)
        vid_score.update(val=id_dis)
    # print(vid_score.avg)
    fake_var.append(np.var(fake_id))
    fake_diff.append(vid_score.avg)

    real_id = []
    vid_id = np.load(os.path.join(cdf_real_root, vid_real))
    vid_score = AverageMeter()
    vid_score.reset()
    length = vid_id.shape[0]
    for i in range(length-1):
        id_dis = face_distance(vid_id[i], vid_id[i+1])
        real_id.append(id_dis)
        vid_score.update(val=id_dis)
    # print(vid_score.avg)
    real_var.append(np.var(real_id))
    real_diff.append(vid_score.avg)



# pdb.set_trace()

x = np.arange(0,int(len(fake_list)))
# label=['Origin Videos', 'DeepFake Videos']
# # plt.plot(x, real_diff, 'g', x, fake_diff, 'r', x, (np.array(fake_diff)-np.array(real_diff)), 'y', x, np.zeros(len(fake_list)), 'b')
# plt.plot(x, real_diff, 'g', x, fake_diff, 'r')
# # plt.title('Identity ')
# # plt.legend(loc="upper right")
# plt.legend(label)
# plt.xlabel('Videos')
# plt.ylabel('Average Identity Distance')
# # plt.ylim(0,10)
# plt.savefig('./images/df_org_id_align_100.pdf')

plt.figure(figsize=(6, 3.8))
colors = ['green', 'red']
label=['Original Videos', 'DeepFake Videos']
n_bins=40
real_diff = np.array(real_diff)
fake_diff = np.array(fake_diff)
diff=[real_diff, fake_diff]

n, bins, patches = plt.hist(diff, n_bins, histtype='bar', density=True, color=colors, label=label)
import pdb
# pdb.set_trace()
# print(n[0])
# print(n[1])
bins = bins[:-1]
coeff_real = polyfit(bins, n[0], 6)
coeff_fake = polyfit(bins, n[1], 5)
y_real = poly1d(coeff_real)(bins)
y_fake = poly1d(coeff_fake)(bins)
plt.plot(bins[0:24], y_real[0:24], 'g--',)
plt.plot(bins[8:], y_fake[8:], 'r--',)
# plt.hist(real_diff, bins=40, facecolor="red", edgecolor="black", alpha=0.8)
# plt.hist(fake_diff, bins=40, facecolor="green", edgecolor="black", alpha=0.8)
plt.legend(loc="upper right")
plt.xlabel('Identity Distance')
plt.ylabel('Frequency')
plt.title('Histogram of Identity Distance')
plt.savefig('./images/df_org_id_align_100_hist.png')

import cv2
img = cv2.imread('./images/df_org_id_align_100_hist.png')

# # pdb.set_trace()

img = cv2.resize(img, (600, 300))

cv2.imwrite('./images/df_org_id_align_100_hist.png', img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

# plt.figure()
# plt.plot(x, real_var, 'g', x, fake_var, 'r', x, (np.array(fake_var)-np.array(real_var)), 'y', x, np.zeros(len(fake_list)), 'b')

# plt.legend(label)
# plt.xlabel('Videos')
# ylabel('')
# plt.savefig('./images/df_org_id_align_200_var.png')

