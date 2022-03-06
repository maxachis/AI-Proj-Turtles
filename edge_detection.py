from imutils import paths
import cv2
import os

path = "./turtles-origcrop/"
new_path = './turtle_edge/'

if not os.path.isdir('turtle_edge'):
    os.mkdir('turtle_edge')

for imagePath in paths.list_images(path):
    # load the image, convert it to grayscale
    image = cv2.imread(imagePath)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)  # Canny Edge Detection
    filename = imagePath.replace(path, "")
    cv2.imwrite(new_path + filename, edges)