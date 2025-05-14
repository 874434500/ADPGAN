import os
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import h5py
from sklearn.feature_extraction.image import extract_patches_2d
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.io as sio

patch_h, patch_w = 32, 32

# List all files
pascal_path = './images/pascal/JPEGImages'
f_list = [f for f in listdir(pascal_path) if isfile(join(pascal_path, f)) and f.endswith('.jpg')]

# Save
all_patches = []
for f in tqdm(f_list):
    im = cv2.imread(os.path.join(pascal_path, f))
    im_h, im_w, _ = im.shape
    max_patches = (im_h * im_w) // (patch_h * patch_w)
    patches = extract_patches_2d(im, (patch_h, patch_w), max_patches=max_patches)
    all_patches.append(patches)

# # List all files
# ImageNet_path = 'D:/xiazai/imagenet/train'
# f_list = [f for f in listdir(ImageNet_path ) if isfile(join(ImageNet_path , f)) and f.endswith('.JPEG')]
#
# for f in tqdm(f_list):
#     im = cv2.imread(os.path.join(ImageNet_path , f))
#     im_h, im_w, _ = im.shape
#     max_patches = (im_h * im_w) // (patch_h * patch_w)
#     patches = extract_patches_2d(im, (patch_h, patch_w), max_patches=max_patches)
#     all_patches.append(patches)



concat_patches = np.concatenate(all_patches, axis=0)
print(f"Total patches shape: {concat_patches.shape}")

print('Saving hdf5 file...')
with h5py.File('./images/pascal/RGB/image_patches.h5', 'w') as f:
    f.create_dataset('patches', data=concat_patches)
print('[!] HDF5 file is ready.')

# Load and continue
with h5py.File('./images/pascal/RGB/image_patches.h5', 'r') as f:
    x_train = f['patches'][:]

# ... 后续代码保持不变
means = []
img_std = []
smooth_img_std = []

num_img_per_bin = 9000
max_std = 110
n_bins = max_std + 1
selected_img = []
std_hist = np.zeros(n_bins)

c = 0
for i in tqdm(range(x_train.shape[0])):
    cur_std = np.std(x_train[i])
    if cur_std > max_std:
        cur_std = max_std
    cur_bin_idx = int(np.floor(cur_std))

    if std_hist[cur_bin_idx] < num_img_per_bin:
        std_hist[cur_bin_idx] += 1
        selected_img.append(x_train[i])
        img_std.append(cur_std)

plt.plot(std_hist)
plt.figure()
plt.hist(img_std)
plt.show()

# Save new dataset
print('Saving mat file...')
train_imgs = np.array(selected_img)
sio.savemat('./images/pascal/RGB/image_resampled.mat', {'patches': train_imgs})

print('[!] Second Mat file is ready.')
