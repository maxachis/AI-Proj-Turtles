from imutils import paths
import argparse
import cv2
import os
def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()
# construct the argument parse and parse the arguments

threshold = 100
if not os.path.isdir('turtle_noblur'):
    os.mkdir('turtle_noblur')
path = "./turtles-origcrop/"
noblur_path = './turtle_noblur/'
for imagePath in paths.list_images(path):
    # load the image, convert it to grayscale
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    # if the focus measure is less than the supplied threshold,
    # then the image should be considered "blurry"
    if fm > threshold:
        #Otherwise, save to new "turtles-noblur" folder
        filename = imagePath.replace(path, "")
        cv2.imwrite(noblur_path+filename, image)