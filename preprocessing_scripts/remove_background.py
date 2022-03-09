import os
from imutils import paths
from rembg.bg import remove
import numpy as np
import io
from PIL import Image
from PIL import ImageFile

path = "../image_datasets/turtles-origcrop/"
new_path = '../image_datasets/turtle_nobg/'

#If "turtle_classes" folder doesn't exist, create it
if not os.path.isdir('../image_datasets/turtle_nobg'):
    os.mkdir('../image_datasets/turtle_nobg')

# Uncomment the following lines if working with trucated image formats (ex. JPEG / JPG)
# In my case I do give JPEG images as input, so i'll leave it uncommented
ImageFile.LOAD_TRUNCATED_IMAGES = True

for imagePath in paths.list_images(path):
    f = np.fromfile(imagePath)
    result = remove(f)
    img = Image.open(io.BytesIO(result)).convert("RGBA")
    filename = imagePath.replace(path, "").replace(".JPG",".PNG" )
    dest = new_path + filename
    img.save(dest)

