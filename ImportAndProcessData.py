from track_with_mediapipe import detect_hand
from kaggle.api.kaggle_api_extended import KaggleApi
import os.path
import cv2
import csv

#api = KaggleApi()
#api.authenticate()

#This file is only inteded to be run once. It will download a dataset from https://www.kaggle.com/datasets/kuzivakwashe/significant-asl-sign-language-alphabet-dataset
#Then it will take those images and convert all of them to binary black and white images, and store them in /Processed_Images/
#The paths to these images and their label will be stored in a csv file for training later


#Downloading the main ASL Dataset, this requires a kaggle account and api key. Details: https://github.com/Kaggle/kaggle-api
#ONLY RUN THIS ONCE, IT DOWNLOADS A 4 GB FILE

#api.dataset_download_files("kuzivakwashe/significant-asl-sign-language-alphabet-dataset")

#Recursively iterate through all directories and files in the given directory, converting all pictures to black and white

read = "significant-asl-alphabet-training-set\Training Set"
read_test = "test_path"
write = "processed_images"


def get_all_images(read_dir, write_dir, parent):
    i = 0
    with open('image_list.csv', mode='a', newline='') as image_list:
        writer = csv.writer(image_list, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for file in os.listdir(read_dir):
            i += 1
            full_path = os.path.join(read_dir, file)
            # checking if it is a file or directory
            if os.path.isfile(full_path):
                output = detect_hand(full_path, 64)
                if output is not None:
                    result_path = "%s\%s%d.jpg" % (write_dir, parent, i)
                    print(result_path)
                    cv2.imwrite(result_path, output)
                    writer.writerow([parent, result_path])
                else:
                    #result_path = "%s\%s%d.jpg" % (write_dir, parent, i)
                    print("Failure at %s" % full_path)

            elif os.path.isdir(full_path):
                get_all_images(full_path, write_dir, file)


get_all_images(read_test, write, read_test)

#process_image("significant-asl-alphabet-training-set\Training Set\A\color_0_0002 (2).png", True)