import os
import glob
import random

# define data root directory.
# Images were saved as /path_to_data_root/case_id/image_patches_extracted_from_this_case
data_root = "/infodev1/non-phi-data/junjiang/OvaryCancer/auto_enc_patches_256"

# define negative cases
borderline_case_id_list = ["OCMC-016", "OCMC-017", "OCMC-018", "OCMC-019", "OCMC-020",
                           "OCMC-021", "OCMC-022", "OCMC-023", "OCMC-024", "OCMC-025",
                           "OCMC-026", "OCMC-027", "OCMC-028", "OCMC-029", "OCMC-030"]
# define positive cases
high_grade_case_id_list = ["OCMC-001", "OCMC-002", "OCMC-003", "OCMC-004", "OCMC-005",
                           "OCMC-006", "OCMC-007", "OCMC-008", "OCMC-009", "OCMC-010",
                           "OCMC-011", "OCMC-012", "OCMC-013", "OCMC-014", "OCMC-015"]

# define cases for training (include both positive and negative)
training_case_id_list = ["OCMC-001", "OCMC-002", "OCMC-003", "OCMC-004", "OCMC-005",
                         "OCMC-006", "OCMC-007", "OCMC-008", "OCMC-009", "OCMC-010"]

# define cases for testing (include both positive and negative)
testing_case_id_list = ["OCMC-011", "OCMC-012", "OCMC-013", "OCMC-014", "OCMC-015"]


# define labels and class text
labels = ["SBOT", "HGSOC"]
class_maps = {0: borderline_case_id_list, 1: high_grade_case_id_list}

# get class label and label text from case ID
def get_class_labels_from_case(case, class_maps, labels):
    for key, value in class_maps.items():
        if case in value:
            label = key
            class_txt = labels[key]
            return label, class_txt
    raise Exception("Unable to get the class label and label text")

'''
###################### create csv file ############################
create csv file, in which all the image patches are included and shuffled
'''
def save_csv_for_tfrecord(save_to, split, data_root, case_id_list, shuffle=True):
    header = "split,image_url,label\n"
    ext = ".jpg"
    temp_save_to = os.path.split(save_to)[1].replace(".csv", "_tmp.csv")
    if not os.path.exists(save_to):
        fpt = open(temp_save_to, 'w')
        for case in case_id_list:
            l, class_txt = get_class_labels_from_case(case, class_maps, labels)
            f_list = glob.glob(os.path.join(data_root, case, "*" + ext))
            wrt_str = ""
            for f in f_list:
                wrt_str += split + "," + f + "," + class_txt + "\n"
            fpt.write(wrt_str)
        fpt.close()

        if shuffle:
            lines = open(temp_save_to).readlines()
            random.shuffle(lines)
            fp = open(save_to, 'w')
            fp.write(header)
            fp.writelines(lines)
            fp.close()
            os.remove(temp_save_to)
        else:
            os.rename(temp_save_to, save_to)
    else:
        print("file has already exists")


'''
# create csv, each line consists of split, image path and label.
Example 1:
split,image_url,label 
TRAIN,/temp/wsi/patches/OCMC-001/OCMC-001_1000_1200.jpg,HGSOC

Example 2:
split,image_url,label 
TEST,/temp/wsi/patches/OCMC-021/OCMC-021_1700_1400.jpg,SBOT
'''

# #save training data
training_save_to = "/infodev1/non-phi-data/junjiang/OvaryCancer/AutoEncoder_Result/training_tf.csv"
split = "TRAIN"
print("Prepare CSV file for saving tfRecord. Training Data")
save_csv_for_tfrecord(training_save_to, split, data_root, training_case_id_list)

'''
###################### create data generator directly from image folder############################
'''
import pandas as pd
import tensorflow as tf
import numpy as np

IMG_SHAPE = (256, 256, 3)
num_classes = len(labels)

header = "split,image_url,label\n"
names = header.strip().split(",")
df = pd.read_csv(training_save_to, names=names)

file_list = df[names[1]].tolist()[1:]
label_txt_list = df[names[2]].tolist()[1:]


def label_to_int(labels_list, class_index):  # labels is a list of labels
    int_classes = []
    for label in labels_list:
        int_classes.append(class_index.index(label))  # the class_index.index() looks values up in the list label
    int_classes = np.array(int_classes, dtype=np.int32)
    return int_classes, class_index  # returning class index so you know what things are

label_list, _ = label_to_int(label_txt_list, labels)

# create a dataset returning slices of `filenames`
dataset = tf.data.Dataset.from_tensor_slices((file_list, label_list))

# parse every image in the dataset using `map`
def _parse_function(filename, label):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32) / 255.
    image = tf.reshape(image, IMG_SHAPE)
    label = tf.cast(tf.one_hot(label, num_classes), tf.int64)
    return image, label

dataset = dataset.map(_parse_function)
dataset = dataset.batch(2)

