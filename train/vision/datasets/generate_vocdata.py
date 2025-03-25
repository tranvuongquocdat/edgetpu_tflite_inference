import glob
import sys
import os
import xml.etree.ElementTree as ET
from random import random

def main(filename, data_folder):
    # ratio to divide up the images
    train = 0.8
    val = 0.15
    test = 0.05
    if (train + test + val) != 1.0:
        print("probabilities must equal 1")
        exit()

    # get the labels
    labels = []
    imgnames = []
    annotations = {}

    with open(filename, 'r') as labelfile:
        label_string = ""
        for line in labelfile:
                label_string += line.rstrip()

    labels = label_string.split(',')
    labels  = [elem.replace(" ", "") for elem in labels]

    # get image names - updated path
    for filename in os.listdir(os.path.join(data_folder, "JPEGImages")):
        if filename.endswith(".jpg"):
            img = filename.rstrip('.jpg')
            imgnames.append(img)

    print("Labels:", labels, "imgcnt:", len(imgnames))

    # initialise annotation list
    for label in labels:
        annotations[label] = []

    # Scan the annotations for the labels - updated path
    for img in imgnames:
        annote = os.path.join(data_folder, "Annotations", img + '.xml')
        if os.path.isfile(annote):
            tree = ET.parse(annote)
            root = tree.getroot()
            annote_labels = []
            for labelname in root.findall('*/name'):
                labelname = labelname.text
                annote_labels.append(labelname)
                if labelname in labels:
                    annotations[labelname].append(img)
            annotations[img] = annote_labels
        else:
            print("Missing annotation for ", annote)
            exit() 

    # divvy up the images to the different sets
    sampler = imgnames.copy()
    train_list = []
    val_list = []
    test_list = []

    while len(sampler) > 0:
        dice = random()
        elem = sampler.pop()

        if dice <= test:
            test_list.append(elem)
        elif dice <= (test + val):
            val_list.append(elem)
        else:
            train_list.append(elem) 

    print("Training set:", len(train_list), "validation set:", len(val_list), "test set:", len(test_list))


    # create the dataset files - updated paths
    image_sets_main = os.path.join(data_folder, "ImageSets", "Main")
    create_folder(image_sets_main)
    
    with open(os.path.join(image_sets_main, "train.txt"), 'w') as outfile:
        for name in train_list:
            outfile.write(name + "\n")
    with open(os.path.join(image_sets_main, "val.txt"), 'w') as outfile:
        for name in val_list:
            outfile.write(name + "\n")
    with open(os.path.join(image_sets_main, "trainval.txt"), 'w') as outfile:
        for name in train_list:
            outfile.write(name + "\n")
        for name in val_list:
            outfile.write(name + "\n")
    with open(os.path.join(image_sets_main, "test.txt"), 'w') as outfile:
        for name in test_list:
            outfile.write(name + "\n")

    # create the individual files for each label - updated paths
    for label in labels:
        with open(os.path.join(image_sets_main, label + "_train.txt"), 'w') as outfile:
            for name in train_list:
                if label in annotations[name]:
                    outfile.write(name + " 1\n")
                else:
                    outfile.write(name + " -1\n")
        with open(os.path.join(image_sets_main, label + "_val.txt"), 'w') as outfile:
            for name in val_list:
                if label in annotations[name]:
                    outfile.write(name + " 1\n")
                else:
                    outfile.write(name + " -1\n")
        with open(os.path.join(image_sets_main, label + "_test.txt"), 'w') as outfile:
            for name in test_list:
                if label in annotations[name]:
                    outfile.write(name + " 1\n")
                else:
                    outfile.write(name + " -1\n")

def create_folder(foldername):
    if os.path.exists(foldername):
        print('folder already exists:', foldername)
    else:
        os.makedirs(foldername)

if __name__=='__main__':
    if len(sys.argv) < 3:
        print("usage: python generate_vocdata.py <labelfile> <data_folder>")
        exit()
    main(sys.argv[1], sys.argv[2])
