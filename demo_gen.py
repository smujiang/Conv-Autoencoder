import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np

from model import create_model
from sklearn.metrics import mean_squared_error

from wsi_data_utils import patch_data_generator, save_all_img_names_and_labels

if __name__ == '__main__':
    img_rows, img_cols = 256, 256
    channel = 3
    batch_sz = 16

    model_weights_path = '/infodev1/non-phi-data/junjiang/Autoencoder_plus/models/model.24-0.0449.hdf5'
    model = create_model()
    #
    model.load_weights(model_weights_path)
    print(model.summary())

    borderline_case_id_list = ["OCMC-016", "OCMC-017", "OCMC-018", "OCMC-019", "OCMC-020",
                               "OCMC-021", "OCMC-022", "OCMC-023", "OCMC-024", "OCMC-025",
                               "OCMC-026", "OCMC-027", "OCMC-028", "OCMC-029", "OCMC-030"]
    high_grade_case_id_list = ["OCMC-001", "OCMC-002", "OCMC-003", "OCMC-004", "OCMC-005",
                               "OCMC-006", "OCMC-007", "OCMC-008", "OCMC-009", "OCMC-010",
                               "OCMC-011", "OCMC-012", "OCMC-013", "OCMC-014", "OCMC-015"]

    borderline_test = ["OCMC-016", "OCMC-017", "OCMC-018", "OCMC-019", "OCMC-020",
                       "OCMC-021", "OCMC-022", "OCMC-023", "OCMC-024", "OCMC-025",
                       "OCMC-026", "OCMC-027", "OCMC-028", "OCMC-029", "OCMC-030"]

    high_grade_test = ["OCMC-001", "OCMC-002", "OCMC-003", "OCMC-004", "OCMC-005",
                       "OCMC-006", "OCMC-007", "OCMC-008", "OCMC-009", "OCMC-010",
                       "OCMC-011", "OCMC-012", "OCMC-013", "OCMC-014", "OCMC-015"]

    class_maps = {0: borderline_case_id_list, 1: high_grade_case_id_list}

    h_img_fn_txt = "/infodev1/non-phi-data/junjiang/OvaryCancer/PatchAEClassification/all_high_grade.txt"
    b_img_fn_txt = "/infodev1/non-phi-data/junjiang/OvaryCancer/PatchAEClassification/all_borderline.txt"

    patch_root = "/infodev1/non-phi-data/junjiang/OvaryCancer/auto_enc_patches_256"
    save_all_img_names_and_labels(h_img_fn_txt, patch_root, high_grade_test, class_maps)
    save_all_img_names_and_labels(b_img_fn_txt, patch_root, borderline_test, class_maps)

    batch_cnt = 0

    test_type_list = ["Borderline", "HighGrade"]
    for test_type in test_type_list:
        if test_type is "HighGrade":
            generator = patch_data_generator(h_img_fn_txt, batch_size=batch_sz)
            testing_eval_csv = "/infodev1/non-phi-data/junjiang/Autoencoder_plus/data_log/MSE_eval_highgrade.csv"
        elif test_type is "Borderline":
            generator = patch_data_generator(b_img_fn_txt, batch_size=batch_sz)
            testing_eval_csv = "/infodev1/non-phi-data/junjiang/Autoencoder_plus/data_log/MSE_eval_borderline.csv"
        else:
            raise Exception("undefined evaluation type")

        # x_test = np.empty((16, img_rows, img_cols, 3), dtype=np.float32)
        rgb_img_batch = next(generator)
        x_test = np.array(rgb_img_batch[0]).astype(np.float32) / 255.0

        out = model.predict(x_test, batch_size=batch_sz)



        # randomly save some examples
        k = random.randrange(1, 100)
        if k == 1:
            for b in range(batch_sz):



        # error_list_str = "img_input,img_output,error\n"
        # for i in range(len(samples)):
        #     filename = samples[i].strip()
        #     # filename = os.path.join(test_path, image_name)
        #
        #     print('Start processing image: {}'.format(filename))
        #
        #     x_test = np.empty((1, img_rows, img_cols, 3), dtype=np.float32)
        #     bgr_img = cv.imread(filename)
        #     rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        #     rgb_img = rgb_img / 255.0
        #     x_test[0, :, :, :] = rgb_img
        #     out = model.predict(x_test, batch_size=)
        #     # print(out.shape)
        #     out = np.squeeze(out)
        #     # out = np.reshape(out, (img_rows, img_cols))
        #     out = out * 255.0
        #
        #     img_fn = os.path.split(filename)[1]
        #     input_fn = 'images/' + img_fn.replace(".jpg", "_input.jpg")
        #     output_fn = 'images/' + img_fn.replace(".jpg", "_output.jpg")
        #
        #     mse = (np.square(rgb_img - out)).mean(axis=None)
        #     error_list_str += input_fn + "," + output_fn + "," + str(mse)+"\n"
        #
        #     out = out.astype(np.uint8)
        #
        #     bgr_out = cv.cvtColor(out, cv.COLOR_RGB2BGR)
        #
        #     cv.imwrite(input_fn, bgr_img)
        #     cv.imwrite(output_fn, bgr_out)
        #
        # fp = open(testing_eval_csv, 'w')
        # fp.write(error_list_str)
        # fp.close()

    K.clear_session()
