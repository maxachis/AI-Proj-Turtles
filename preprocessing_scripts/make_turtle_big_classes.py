# This file will create a dataset that consists of multiple variants of each image, preprocessed in different ways, in a class format

import os
import shutil
import pandas as pd

src_directories = ["turtle_edge", "turtle_nobg", "turtle_noblur", "turtles-origcrop"]

def main():
    train = pd.read_csv('../train.csv')

    #If "turtle_classes" folder doesn't exist, create it
    if not os.path.isdir('../image_datasets/turtle_big_classes'):
        os.mkdir('../image_datasets/turtle_big_classes')

    #For each row in train
    for row in train.iterrows():
        turtle_id = row[1]['turtle_id']
        image_id = row[1]['image_id']
        if not os.path.isdir('../image_datasets/turtle_big_classes/' + turtle_id):
            os.mkdir('../image_datasets/turtle_big_classes/' + turtle_id)
        #Get file
        for dir in src_directories:
            src = "../image_datasets/" + dir + "/" + image_id + '.jpg'
            if os.path.exists(src):
                dst = '../image_datasets/turtle_big_classes/' + turtle_id + '/' + image_id + '_' + dir + '.jpg'
                shutil.copyfile(src, dst)

main()