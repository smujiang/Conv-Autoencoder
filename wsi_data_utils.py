import os
import glob
import numpy as np
from PIL import Image
from itertools import cycle
#
def save_all_img_names_and_labels(save_to, data_root, case_list, class_maps, ext=".jpg"):
    '''
    List all the name of image patches from cases which are listed in case_list
    list all the class labels for the images
    then save them to a txt file, format: each row is an example, consists of label and image file path
    :param save_to:  file name where the data will save to
    :param data_root: root dir to the image files
    :param case_list:  example, case_list = ["OCMC-016", "OCMC-017", "OCMC-001", "OCMC-002"]
    :param class_maps: example, class_maps = {0: ["OCMC-016", "OCMC-017"], 1: ["OCMC-001", "OCMC-002"]}
    :param ext: extension of the image file name
    :return:
    '''
    wrt_str = ""
    if not os.path.exists(save_to):
        fp = open(save_to, 'a')
        for case in case_list:
            f_list = glob.glob(os.path.join(data_root, case, "*"+ext))
            label = "0"
            for key, value in class_maps.items():
                if case in value:
                    label = str(key)
                    break
            for f in f_list:
                wrt_str += label + "," + f + "\n"
            fp.write(wrt_str)
        fp.close()
    else:
        print("file has already exists")

def list_all_img_names(data_root, case_list, ext=".jpg"):
    all_f_list = []
    for case in case_list:
        f_list = glob.glob(os.path.join(data_root, case, "*"+ext))
        all_f_list += f_list
    return all_f_list

def patch_data_generator(img_fn_list, batch_size, mode="train"):
    img_total_cnt = len(img_fn_list)
    img_idx = 0
    pool = cycle(img_fn_list)
    while True:
        images = []
        while len(images) < batch_size:
            img_fn = next(pool)
            if img_idx == img_total_cnt:
                img_idx = 0
                if mode == "eval":  # don't need to repeat several rounds for evaluation
                    break
            image = np.array(Image.open(img_fn, 'r'))
            images.append(image)
            img_idx += 1
        yield np.array(images)


def patch_data_label_generator(data_list_txt, batch_size, mode="train", aug=None):
    f = open(data_list_txt, "r")
    while True:
        images = []
        labels = []
        while len(images) < batch_size:
            line = f.readline()
            if line == "":
                f.seek(0)  # go to the beginning of the file
                line = f.readline()
                if mode == "eval":  # don't need to repeat several rounds for evaluation
                    break
            line = line.strip().split(",")
            label = int(line[0])
            image = np.array(Image.open(line[1], 'r'))
            images.append(image)
            labels.append(label)
        if aug is not None:
            (images, labels) = next(aug.flow(np.array(images), labels, batch_size=batch_size))
        yield (np.array(images), labels)



if __name__ == "__main__":
    patch_root = "/infodev1/non-phi-data/junjiang/OvaryCancer/auto_enc_patches_256"
    case_list = ["OCMC-016", "OCMC-001"]

    # borderline_case_id_list = ["OCMC-016"]
    # high_grade_case_id_list = ["OCMC-001"]

    borderline_case_id_list = ["OCMC-016", "OCMC-017", "OCMC-018", "OCMC-019", "OCMC-020",
                               "OCMC-021", "OCMC-022", "OCMC-023", "OCMC-024", "OCMC-025",
                               "OCMC-026", "OCMC-027", "OCMC-028", "OCMC-029", "OCMC-030"]
    high_grade_case_id_list = ["OCMC-001", "OCMC-002", "OCMC-003", "OCMC-004", "OCMC-005",
                               "OCMC-006", "OCMC-007", "OCMC-008", "OCMC-009", "OCMC-010",
                               "OCMC-011", "OCMC-012", "OCMC-013", "OCMC-014", "OCMC-015"]

    borderline_train = ["OCMC-016", "OCMC-017", "OCMC-018", "OCMC-019", "OCMC-020",
                        "OCMC-021", "OCMC-022", "OCMC-023", "OCMC-024"]
    borderline_validate = ["OCMC-025", "OCMC-026", "OCMC-027"]
    borderline_test = ["OCMC-028", "OCMC-029", "OCMC-030"]

    high_grade_train = ["OCMC-001", "OCMC-002", "OCMC-003", "OCMC-004", "OCMC-005",
                        "OCMC-006", "OCMC-007", "OCMC-008", "OCMC-009"]
    high_grade_validate = ["OCMC-010", "OCMC-011", "OCMC-012"]
    high_grade_test = ["OCMC-013", "OCMC-014", "OCMC-015"]

    class_maps = {0: borderline_case_id_list, 1: high_grade_case_id_list}

    data_train_txt = os.path.join(patch_root, "training.txt")
    data_validate_txt = os.path.join(patch_root, "validation.txt")
    data_test_txt = os.path.join(patch_root, "testing.txt")

    '''
    Test 1:
    '''
    # all_list = list_all_img_names(patch_root, case_list, ext=".jpg")
    # gen = patch_data_generator(all_list, batch_size=16, mode='eval')
    # print(next(gen))

    '''
    Test 2:
    '''
    save_all_img_names_and_labels(data_train_txt, patch_root, borderline_train+high_grade_train, class_maps)
    save_all_img_names_and_labels(data_validate_txt, patch_root, borderline_validate + high_grade_validate, class_maps)
    save_all_img_names_and_labels(data_test_txt, patch_root, borderline_test + high_grade_test, class_maps)


    # test function
    '''
    d = patch_data_label_generator(data_test_txt, batch_size=2)

    # validate the data use print
    print(next(d)[0])
    print(next(d)[1])

    print(d.__next__())
    
    '''






















