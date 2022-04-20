def imageResizeTest(image):
    maxD = 224
    height,width,channel = image.shape
    aspectRatio = width/height
    if aspectRatio < 1:
        newSize = (int(maxD*aspectRatio),maxD)
    else:
        newSize = (maxD,int(maxD/aspectRatio))
    image = cv2.resize(image,newSize)

    # trim the edges of the image
    trim = 17
    h,w,c = image.shape
    image = image[-h+trim:h-trim,-w+trim:w-trim]

    return image
