import os
import shutil
import pandas as pd

def main():
    train = pd.read_csv('train.csv')

    #If "turtle_classes" folder doesn't exist, create it
    if not os.path.isdir('turtle_classes'):
        os.mkdir('turtle_classes')

    #For each row in train
    for row in train.iterrows():
        turtle_id = row[1]['turtle_id']
        image_id = row[1]['image_id']
        if not os.path.isdir('turtle_classes/' + turtle_id):
            os.mkdir('turtle_classes/' + turtle_id)
        #Get file
        src = 'turtles-origcrop/'+ image_id + '.jpg'
        if os.path.exists(src):
            dst = 'turtle_classes/' + turtle_id + '/' + image_id + '.jpg'
            shutil.copyfile(src, dst)