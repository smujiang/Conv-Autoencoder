import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np

from model import create_model
from sklearn.metrics import mean_squared_error

from wsi_data_utils import list_all_img_names

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
    data_root = "/infodev1/non-phi-data/junjiang/OvaryCancer/auto_enc_patches_256"
    data_out_root = "/infodev1/non-phi-data/junjiang/OvaryCancer/PatchAEClassification/reconstructed"

    all_cases = borderline_case_id_list+high_grade_case_id_list

    for c in all_cases:
        test_images = list_all_img_names(data_root, [c])

        testing_eval_csv = os.path.join(data_out_root, c+".csv")
        samples = random.sample(test_images, 100)
        error_list_str = "img_input,img_output,error\n"
        for i in range(len(samples)):
            filename = samples[i].strip()
            # filename = os.path.join(test_path, image_name)

            print('Start processing image: {}'.format(filename))

            x_test = np.empty((1, img_rows, img_cols, 3), dtype=np.float32)
            bgr_img = cv.imread(filename)
            rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
            rgb_img = rgb_img / 255.0
            x_test[0, :, :, :] = rgb_img
            out = model.predict(x_test)
            # print(out.shape)
            out = np.squeeze(out)
            # out = np.reshape(out, (img_rows, img_cols))
            out = out * 255.0

            img_fn = os.path.split(filename)[1]
            input_fn = os.path.join(data_out_root, c, img_fn.replace(".jpg", "_input.jpg"))
            output_fn = os.path.join(data_out_root, c, img_fn.replace(".jpg", "_output.jpg"))
            if not os.path.exists(os.path.join(data_out_root, c)):
                os.makedirs(os.path.join(data_out_root, c))
            mse = (np.square(rgb_img - out)).mean(axis=None)
            error_list_str += input_fn + "," + output_fn + "," + str(mse) + "\n"

            out = out.astype(np.uint8)

            bgr_out = cv.cvtColor(out, cv.COLOR_RGB2BGR)

            cv.imwrite(input_fn, bgr_img)
            cv.imwrite(output_fn, bgr_out)

        fp = open(testing_eval_csv, 'w')
        fp.write(error_list_str)
        fp.close()
    K.clear_session()
