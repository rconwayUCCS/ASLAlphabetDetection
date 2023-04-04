from track_with_mediapipe import detect_hand
from kaggle.api.kaggle_api_extended import KaggleApi
import numpy as np
import os.path
import cv2
import csv

#api = KaggleApi()
#api.authenticate()

#This file is only inteded to be run once. It will download a dataset from https://www.kaggle.com/datasets/kuzivakwashe/significant-asl-sign-language-alphabet-dataset
#Then it will take those images and convert all of them to binary black and white images, and store them in /Processed_Images/
#Other dataset https://www.kaggle.com/datasets/grassknoted/asl-alphabet
#The paths to these images and their label will be stored in a csv file for training later


#Downloading the main ASL Dataset, this requires a kaggle account and api key. Details: https://github.com/Kaggle/kaggle-api
#ONLY RUN THIS ONCE, IT DOWNLOADS A 4 GB FILE

#api.dataset_download_files("kuzivakwashe/significant-asl-sign-language-alphabet-dataset")

#Recursively iterate through all directories and files in the given directory, converting all pictures to black and white
#The paths to these images and their labels are stored in a csv file

read = "TrainingSet2"
read_test = "test_path"
write = "processed_images_2"
write_csv = 'image_list2.csv'


def get_all_images(read_dir, write_dir, parent):
    i = 0
    failures = 0
    with open(write_csv, mode='a', newline='') as image_list:
        writer = csv.writer(image_list, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        print("Processing ", read_dir)
        for file in os.listdir(read_dir):
            full_path = os.path.join(read_dir, file)
            # checking if it is a file or directory
            if os.path.isfile(full_path):
                output = detect_hand(full_path, 64)
                if output is not None:
                    result_path = "%s\%s%d.jpg" % (write_dir, parent, i)
                    #print(result_path)
                    cv2.imwrite(result_path, output)
                    #print(output.shape)
                    #np.savetxt(write_csv, (np.int_(output//255)))
                    writer.writerow([parent, result_path])
                    i += 1
                else:
                    failures += 1

            elif os.path.isdir(full_path):
                get_all_images(full_path, write_dir, file)
        print("%d successes, %d failures in %s" % (i, failures, parent))

with open(write_csv, mode='w', newline='') as image_list:
   writer = csv.writer(image_list, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

get_all_images(read, write, read)