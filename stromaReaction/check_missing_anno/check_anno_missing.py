import os
import glob
from itertools import combinations
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

all_class_list = ["Fibrosis", "Cellularity", "Orientation"]
testing_cases = ["Wo-1-A5_RIO1338_HE", "Wo-1-C4_RIO1338_HE", "Wo-1-F1_RIO1338_HE", "Wo-2-B4_RIO1338_HE",
                 "Wo-2-F1_RIO1338_HE"]

anno_dir = "/infodev1/non-phi-data/junjiang/OvaryCancer/StromaReaction/Annotation"
out_put_dir = "/infodev1/non-phi-data/junjiang/OvaryCancer/StromaReaction/eval"

def check_mask_img_same_shape(img1_fn, img2_fn):
    img1 = Image.open(img1_fn).convert('L')
    img2 = Image.open(img2_fn).convert('L')
    # img1_arr = np.array(img1)
    threshold = 254
    im1 = img1.point(lambda p: p > threshold and 255)
    im2 = img2.point(lambda p: p > threshold and 255)
    # plt.imshow(img1, cmap="gray")
    # plt.show()
    # plt.imshow(im1, cmap="gray")
    # plt.show()
    result = np.array(im1) - np.array(im2)
    if np.any(result):
        return False
    else:
        return True

txt_fn = os.path.join(out_put_dir, "check_mask_same_shape.txt")
fp = open(txt_fn, "w")
fp.write("location_x,location_y,case_id\n")

for case in testing_cases:
    img_fn_list = glob.glob(os.path.join(anno_dir, case, "*.jpg"))
    for img_fn in img_fn_list:
        mask_fn_list = []
        for cl in all_class_list:
            mask_fn = img_fn.replace(".jpg", "_" + cl + "-mask.png")
            if not os.path.exists(mask_fn):
                raise Exception("mask image not exist")
            mask_fn_list.append(mask_fn)

        # check if all annotation exist
        comb = combinations(mask_fn_list, 2)
        for mask_fn1, mask_fn2 in comb:
            same_shape = check_mask_img_same_shape(mask_fn1, mask_fn2)
            if not same_shape:
                img_fn = os.path.split(img_fn)[1]
                ele = img_fn.split("_")
                loc_x = int(ele[-4])
                loc_y = int(ele[-3])
                fp.write(str(loc_x) + "," + str(loc_y) + "," + case + "\n")
                break
fp.close()














