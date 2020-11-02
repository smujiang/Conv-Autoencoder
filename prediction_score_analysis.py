import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD
import os
import migrate
# from vgg16_original import vgg16_model_org
from vgg16 import vgg16_updated
from utils import custom_loss
from wsi_data_utils import patch_data_label_generator
import glob
from PIL import Image
import numpy as np
import openslide
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img_rows = 256
    img_cols = 256
    img_rescale_rate = 2

    thumbnail_downsample = 128

    # Load our model
    model_weights_path = '/infodev1/non-phi-data/junjiang/Autoencoder_plus/models/unsampled_model.01-1.1702.hdf5'
    model = vgg16_updated()
    model.load_weights(model_weights_path)
    print(model.summary())

    # Load our data
    patch_root = "/infodev1/non-phi-data/junjiang/OvaryCancer/auto_enc_patches_256"
    WSI_root = "/infodev1/non-phi-data/junjiang/OvaryCancer/WSIs"
    output_dir = "/infodev1/non-phi-data/junjiang/OvaryCancer/PatchClassification/VGG16_pilot"

    borderline_test = ["OCMC-028", "OCMC-029", "OCMC-030"]
    high_grade_test = ["OCMC-013", "OCMC-014", "OCMC-015"]

    testing_cases = borderline_test + high_grade_test
    cmap = plt.get_cmap("jet")
    # cnt = 100
    for case in testing_cases:
        print("processing %s " % case)
        eval_out_csv = os.path.join(output_dir, "eval" + case + ".csv")
        csv_fp = open(eval_out_csv, 'w')
        eval_results = "location_x,location_y,width,height,sore1,score2\n"

        f_list = glob.glob(os.path.join(patch_root, case, "*.jpg"))
        score_list = []
        location_list = []
        test_x = np.empty((1, img_rows, img_cols, 3), dtype=np.float32)
        for f in f_list:
            # x = np.array(Image.open(f, 'r')).astype(np.float32)/255.0
            test_x[0, :, :, :] = np.array(Image.open(f, 'r')).astype(np.float32) / 255.0
            out = model.predict(test_x)
            # get score for heat-map
            score_list.append(out[0][0])
            # get locations from image file name
            img_fn = os.path.split(f)[1]
            ele = img_fn.split("_")
            location_list.append([int(ele[1]), int(ele[2][0:-4])])
            eval_results += str(int(ele[1])) + "," + str(int(ele[2][0:-4])) + "," + str(img_cols * img_rescale_rate) \
                            + "," + str(img_rows * img_rescale_rate) + "," + str(out[0][0]) + "," + str(out[0][1]) + "\n"
            # cnt -= 1
            # if cnt == 0:
            #     break
        csv_fp.write(eval_results)
        csv_fp.close()
        print(score_list.__len__())
        # get thumbnail of WSI
        wsi_fn = os.path.join(WSI_root, case + ".svs")
        wsi_obj = openslide.open_slide(wsi_fn)
        thumb_size = np.array(wsi_obj.dimensions) / thumbnail_downsample
        thumb_img = wsi_obj.get_thumbnail(thumb_size)
        plt.imshow(thumb_img)
        plt.axis("off")
        plt.show()

        # overlap heat-map to WSI thumbnail
        #  TODO: refer to this like: https://github.com/smujiang/Cytomine_GetAnnotation/blob/master/validate_exportation.py
        heat_map_img = np.zeros([thumb_img.height, thumb_img.width, 4]).astype(np.uint8)  # TODO: create heat map image

        for idx, loc in enumerate(location_list):
            start_x = int(loc[0] / thumbnail_downsample)
            end_x = int((loc[0] + img_rows * img_rescale_rate) / thumbnail_downsample)
            start_y = int(loc[1] / thumbnail_downsample)
            end_y = int((loc[1] + img_cols * img_rescale_rate) / thumbnail_downsample)

            cmap_val = np.array(np.array(cmap(score_list[idx])) * 255).astype(np.uint8)
            x = np.broadcast_to(cmap_val, (end_y - start_y, end_x - start_x, 4))
            heat_map_img[start_y:end_y, start_x:end_x, :] = x
        plt.imshow(heat_map_img)
        plt.axis("off")
        plt.show()

        heat_map_img = Image.fromarray(heat_map_img)
        if heat_map_img.mode == "RGB":
            a_channel = Image.new('L', heat_map_img.size, 255)  # 'L' 8-bit pixels, black and white
            heat_map_img.putalpha(a_channel)

        if thumb_img.mode == "RGB":
            a_channel = Image.new('L', thumb_img.size, 255)  # 'L' 8-bit pixels, black and white
            thumb_img.putalpha(a_channel)
        blended = Image.blend(thumb_img, heat_map_img, 0.7)
        save_to = os.path.join(output_dir, case + "_blended.png")
        blended.save(save_to)
        save_to = os.path.join(output_dir, case + "_thumb.png")
        thumb_img.save(save_to)
        save_to = os.path.join(output_dir, case + "_heat_map.png")
        heat_map_img.save(save_to)

    #
    # '''
    # # testing data generator
    # '''
    # data_test_txt = os.path.join(patch_root, "testing.txt")
    # test_gen = patch_data_label_generator(data_test_txt, batch_size=batch_size)
    #
    # test_img_fns = open(data_test_txt, "r").readlines()
    # test_cnt = len(test_img_fns)
    #
    # for i in range(int(test_cnt/batch_size)):
    #     rgb_img_batch = next(test_gen)
    #     test_x = rgb_img_batch[0]
    #     test_y = rgb_img_batch[1]
    #     out = model.predict(test_x, batch_size=batch_size)
    #
    #     # create heat-map
    #
    #     test_i
    #

    # prediction
    # prd = model.evaluate_generator(test_gen, int(test_cnt/batch_size))
    # input_fn = tf.estimator.inputs.numpy_input_fn(test_gen)
    # for single_prediction in model.predict(input_fn):
    #     predicted_class = single_prediction['class']
    #     probability = single_prediction['probability']
    # prd = model.predict(test_gen, batch_size=batch_size)
