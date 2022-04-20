# Resize images to a similar dimension
# This helps improve accuracy and decreases unnecessarily high number of keypoints

def resize_image(image, drop_channel=True):
    maxD = 224
    if drop_channel:
        height, width, channel = image.shape
    else
        height, width = image.shape
    aspectRatio = width/height
    if aspectRatio < 1:
        newSize = (int(maxD*aspectRatio),maxD)
    else:
        newSize = (maxD,int(maxD/aspectRatio))
    image = cv2.resize(image,newSize)

    # trim the edges of the image
    trim = 17
    h,w = image.shape
    image = image[-h+trim:h-trim,-w+trim:w-trim]

    return image
