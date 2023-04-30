from track_with_mediapipe import detect_hand
import numpy as np
import os.path
import csv

#Recursively iterate through all directories and files in the given directory, calling track_with_mediapipe() on each image
#The paths to these images and their labels are stored in a csv file

read = "significant-asl-alphabet-training-set\Training Set"
read_test = "test_path"
write = "processed_images_2"
write_csv = 'ArrayTest.csv'

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
                    #result_path = "%s\%s%d.jpg" % (write_dir, parent, i)
                    #print(result_path)
                    #cv2.imwrite(result_path, output)
                    #print(output.shape)
                    #np.savetxt(write_csv, 'a', output)
                    conv_x, conv_y, conv_z = convert_palm_normal(output[0], output[1], output[2])
                    writer.writerow([alphabet.index(parent), *conv_x, *conv_y, *conv_z])
                    i += 1
                else:
                    failures += 1

            elif os.path.isdir(full_path):
                get_all_images(full_path, write_dir, file)
        print("%d successes, %d failures in %s" % (i, failures, parent))


def convert_palm_normal(xs, ys, zs):
    palm_vec1 = [xs[5] - xs[0], ys[5] - ys[0], zs[5] - zs[0]]
    palm_vec2 = [xs[17] - xs[0], ys[17] - ys[0], zs[17] - zs[0]]

    palm_normal = np.cross(palm_vec1, palm_vec2)

    inverse_rho_v = 1 / np.linalg.norm(palm_normal)
    mag_vv = np.linalg.norm([palm_normal[0], palm_normal[1]])

    v3 = palm_normal[2]

    x_primes = []
    y_primes = []
    z_primes = []

    for i in range(len(xs)):
        mag_xy = np.linalg.norm([xs[i], ys[i]])
        xy_coeff = (v3 + ((zs[i]*mag_vv)/mag_xy)) * inverse_rho_v
        z_new = (mag_xy*v3 + zs[i]*mag_vv) * inverse_rho_v

        x_primes.append(xs[i] * xy_coeff)
        y_primes.append(ys[i] * xy_coeff)
        z_primes.append(z_new)

    return x_primes, y_primes, z_primes


#This exists to wipe the file before writing
with open(write_csv, mode='w', newline='') as image_list:
   writer = csv.writer(image_list, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

get_all_images(read_test, write, read_test)