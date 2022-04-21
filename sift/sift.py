import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read in csv files.
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

turtle_imgs_dir = "image_datasets/turtle_origcrop"
train_image_paths = [os.path.join(turtle_imgs_dir, "%s.JPG" % f) for f in train.image_id]
train_images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in train_image_paths]
# TODO: add test images?

sift = cv2.xfeatures2d.SIFT_create(edgeThreshold=5, sigma=1.5, contrastThreshold=0.03, nOctaveLayers=1)
keypoints, descriptors = zip(*[sift.detectAndCompute(image, None) for image in train_images])

# The Image Matcher
bf = cv2.BFMatcher()

a, b = (0, 5)

def find_matches(a, b):
    des1 = descriptors[a]
    des2 = descriptors[b]
    key1 = keypoints[a]
    key2 = keypoints[b]

    matches = bf.knnMatch(des1, des2, k=2)
    matches2 = bf.knnMatch(des2, des1, k=2)

    good = [[m] for (m,n) in matches if m.distance < 0.8 * n.distance]
    good2 = [[m] for (m,n) in matches2 if m.distance < 0.8 * n.distance]
    bimatches = [m1 for m1 in good for m2 in good2 if (m1[0].queryIdx == m2[0].trainIdx) and (m1[0].trainIdx == m2[0].queryIdx)]

    # Filter the matches by removing outlier matches.
    if len(bimatches) < 4: # minimum required to find_homography
        return []

    src_pts = np.float32([key1[m[0].queryIdx].pt for m in bimatches]).reshape(-1, 1, 2)
    dst_pts = np.float32([key2[m[0].trainIdx].pt for m in bimatches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,ransacReprojThreshold=7.0)
    matchesMask = mask.ravel().tolist()
    filtered_matches = [i for i,j in zip(bimatches, matchesMask) if j == 1]

    return filtered_matches

def generate_img(a, b):
    matches = find_matches(a, b)
    return cv2.drawMatchesKnn(train_images[a], keypoints[a], train_images[b], keypoints[b], matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

def match_all_turtles(t_idx):
    all_idx = list(range(len(keypoints)))
    all_idx.remove(t_idx)
    scores = [(len(find_matches(t_idx, i)), i) for i in all_idx]
    scores.sort()
    return scores

def output_top4(t_idx):
    scores = match_all_turtles(t_idx)
    scores.sort(reverse=True, key=lambda x: x[0])
    top4 = scores[0:4]
    fig, axes = plt.subplots(2,2, figsize=(16, 8), tight_layout=True)
    for i, (score, b_idx) in enumerate(top4):
        print(train.turtle_id[t_idx], train.turtle_id[b_idx])
        ax_idx = (i // 2, i % 2)
        im = axes[ax_idx].imshow(generate_img(t_idx, b_idx))
        axes[ax_idx].set_axis_off()
        axes[ax_idx].set_title("LEFT %s / RIGHT %s: score=%d" % (train.turtle_id[t_idx], train.turtle_id[b_idx], score))
    fig.show()
    fig.savefig("%s.png" % train.turtle_id[t_idx])

t_idx = 675
output_top4(t_idx)
