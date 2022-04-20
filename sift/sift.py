import os

import cv2
import matplotlib.pyplot as plt
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





a, b = (0,5)

def plot_matches(a,b):
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
    src_pts = np.float32([ key1[m[0].queryIdx].pt for m in bimatches ]).reshape(-1,1,2)
    dst_pts = np.float32([ key2[m[0].trainIdx].pt for m in bimatches ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,ransacReprojThreshold=7.0)
    matchesMask = mask.ravel().tolist()
    filtered_matches = [i for i,j in zip(bimatches,matchesMask) if j==1]

    print(len(good), len(good2), len(bimatches), len(filtered_matches))

    img3 = cv2.drawMatchesKnn(train_images[a], key1, train_images[b], key2, filtered_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3)
    plt.show()


def calculate_matches(des1, des2, keypoint1, keypoint2, MIN_MATCH_COUNT):
    matches = bf.knnMatch(des1, des2, k=2)
    topResults1 = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            topResults1.append([m])

    matches = bf.knnMatch(des2,des1,k=2)
    topResults2 = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            topResults2.append([m])


    topResults = []
    for match1 in topResults1:
        match1QueryIndex = match1[0].queryIdx
        match1TrainIndex = match1[0].trainIdx

        for match2 in topResults2:
            match2QueryIndex = match2[0].queryIdx
            match2TrainIndex = match2[0].trainIdx

            if (match1QueryIndex == match2TrainIndex) and (match1TrainIndex == match2QueryIndex):
                topResults.append(match1)

    # Filter the matches by removing outlier matches.
    if len(topResults)>MIN_MATCH_COUNT:
        src_pts = np.float32([ keypoint1[m[0].queryIdx].pt for m in topResults ]).reshape(-1,1,2)
        dst_pts = np.float32([ keypoint2[m[0].trainIdx].pt for m in topResults ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,ransacReprojThreshold=7.0)
        matchesMask = mask.ravel().tolist()
        filtered_matches = [i for i,j in zip(topResults,matchesMask) if j==1]
    else:
        filtered_matches = topResults

    return topResults, filtered_matches

def getPlotFor(i,j,keypoint1,keypoint2,matches):
    image1 = imageResizeTest(cv2.imread(IMAGE_DIR+"/"+imageList[i]))
    image2 = imageResizeTest(cv2.imread(IMAGE_DIR+"/"+imageList[j]))
    return getPlot(image1,image2,keypoint1,keypoint2,matches)

def getPlot(image1,image2,keypoint1,keypoint2,matches):
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    matchPlot = cv2.drawMatchesKnn(image1,keypoint1,image2,keypoint2,matches,None,[255,255,255],flags=2)
    return matchPlot

def calculate_score(matches, keypoint1, keypoint2):
    ### if the number of matches found between the two images equals the number of keypoints in each image
    ### then the two imgaes are identical. (100 is the best score)

    return 100 * (matches / min(keypoint1, keypoint2))

def calculateResultsFor(i,j, stats=False,MIN_MATCH_COUNT=3,scorebymatch=False):
    """
    i,j are indices from the image list.
    This function returs a score representing how similar the images at i & j are.
    """

    keypoint1 = keypoints[i]
    descriptor1 = descriptors[i]
    keypoint2 = keypoints[j]
    descriptor2 = descriptors[j]

    # Getting the matches
    matches, filtered_matches = calculateMatches(descriptor1,descriptor2,keypoint1,keypoint2,MIN_MATCH_COUNT)
    return calculateScore(filtered_matches)
    # if stats:
    #     score1 = calculateScore(filtered_matches)
    #     score2 = calculateScore(matches)
    #     plot = getPlotFor(i,j,keypoint1,keypoint2,filtered_matches)
    #     print(f"ALL MATCHES: {len(matches)}\n",f"FILTERED MATCHES: {len(filtered_matches)}\n",
    #           len(keypoint1),len(keypoint2),len(descriptor1),len(descriptor2))
    #     print(f"SCORE FILTERD: {np.round(score1,5)}\n", f"SCORE: {np.round(score2,5)}")
    #     plt.imshow(plot),plt.show()

    #     return score1,score2,filtered_matches

    # if scorebymatch:
    #     ## The more matches found the better
    #     return len(filtered_matches)

    # else:
        score1 = calculateScore(filtered_matches)
        # return score1
