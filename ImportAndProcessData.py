from track_with_mediapipe import detect_hand, convert_palm_normal, normalize
import numpy as np
import os.path
import csv

#Recursively iterate through all directories and files in the given directory, calling track_with_mediapipe() on each image
#The paths to these images and their labels are stored in a csv file

read = "TrainingSet3"
read_test = "test_path"
write = "processed_images_2"
write_csv = 'Dummy.csv'

alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

def get_all_images(read_dir, write_dir, parent):
    i = 0
    failures = 0
    with open(write_csv, mode='a', newline='') as image_list:
        writer = csv.writer(image_list, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        print("Processing ", read_dir)
        for file in os.listdir(read_dir):
            full_path = os.path.join(read_dir, file)
            # checking if it is a file or directory
            if os.path.isfile(full_path):
                output = detect_hand(full_path)
                if output is not None:
                    conv_x, conv_y, conv_z = convert_palm_normal(output[0], output[1], output[2])
                    norm_x, norm_y, norm_z = normalize(conv_x, conv_y, conv_z)
                    writer.writerow([alphabet.index(parent), *norm_x, *norm_y, *norm_z])
                    i += 1
                else:
                    failures += 1

            elif os.path.isdir(full_path):
                get_all_images(full_path, write_dir, file)
        print("%d successes, %d failures in %s" % (i, failures, parent))

print(f"You are about to wipe {write_csv}, type YES to continue")
inp = input()

if inp != "YES":
    exit()

#This exists to wipe the file before writing
with open(write_csv, mode='w', newline='') as image_list:
   writer = csv.writer(image_list, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

get_all_images(read, write, read)

#read_csv = "CoordList.csv"

#with open(read_csv, mode='r', newline='') as image_list:
#    reader = csv.reader(image_list, delimiter=',', quoting=csv.QUOTE_MINIMAL)
#    with open(write_csv, mode='a', newline='') as write_list:
#        writer = csv.writer(write_list, delimiter=',', quoting=csv.QUOTE_MINIMAL)
#
#        raw_x = []
#        raw_y = []
#        raw_z = []
#        for row in reader:
#            label = float(row[0])
#            
#            raw_x = np.array(row[1:22]).astype(float)
#            raw_y = np.array(row[22:43]).astype(float)
#            raw_z = np.array(row[43:]).astype(float)
#
#            norm_x, norm_y, norm_z = normalize(raw_x, raw_y, raw_z)
#
#            writer.writerow([label, *norm_x, *norm_y, *norm_z])
#
#            #data.append(np.array(row[1:]).astype(float))

    