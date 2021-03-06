from tensorflow.keras.applications import VGG16
import os
import glob
from PIL import Image
import numpy as np
import openslide
import matplotlib.pyplot as plt
import natsort


if __name__ == '__main__':
    img_rows = 256
    img_cols = 256
    img_channel = 3
    img_rescale_rate = 2

    thumbnail_downsample = 128
    cmap = plt.get_cmap("jet")
    IMG_SHAPE = (img_rows, img_cols, img_channel)
    num_classes = 3

    all_class_list = ["Fibrosis", "Cellularity", "Orientation"]
    # all_model_list = ["Fibrosis_25-0.1366.hdf5", "Cellularity_18-0.1784.hdf5", "Orientation_27-0.2193.hdf5"]
    all_model_list = ["Fibrosis_10-0.1976.hdf5", "Cellularity_16-0.0911.hdf5", "Orientation_06-0.1932.hdf5"]

    # Load our data
    patch_root = "/infodev1/non-phi-data/junjiang/OvaryCancer/StromaReaction/Testing_ROI_Patches"
    WSI_root = "/infodev1/non-phi-data/junjiang/OvaryCancer/StromaReaction/WSIs"
    output_dir = "/infodev1/non-phi-data/junjiang/OvaryCancer/StromaReaction/VGG16_Classification_ROIs_validation_with_aug"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # testing_cases = ["Wo-2-F5_RIO1338_HE", "Wo-2-L1_RIO1338_HE", "Wo-2-N1_RIO1338_HE",
    #                  "Wo-3-B5_RIO1338_HE", "Wo-3-C1_RIO1338_HE", "Wo-4-A9_RIO1338_HE",
    #                  "Wo-4-B1_RIO1338_HE"]

    # testing_cases = ["Wo-1-A5_RIO1338_HE", "Wo-1-C4_RIO1338_HE", "Wo-1-F1_RIO1338_HE", "Wo-2-B4_RIO1338_HE", "Wo-2-F1_RIO1338_HE"]
    testing_cases = ["Wo-1-A5_RIO1338_HE", "Wo-1-C4_RIO1338_HE", "Wo-1-F1_RIO1338_HE",
                 "Wo-2-B4_RIO1338_HE", "Wo-2-F1_RIO1338_HE",
                 "Wo-2-F5_RIO1338_HE", "Wo-2-L1_RIO1338_HE", "Wo-2-N1_RIO1338_HE",
                 "Wo-3-B5_RIO1338_HE", "Wo-3-C1_RIO1338_HE"]
    # Load our model
    model_weights_path = '/infodev1/non-phi-data/junjiang/OvaryCancer/StromaReaction/model'
    VGG16_MODEL = VGG16(include_top=True, weights=None, input_tensor=None, input_shape=IMG_SHAPE,
                        pooling=None, classes=num_classes, classifier_activation='softmax')
    print(VGG16_MODEL.summary())
    for idx, m in enumerate(all_class_list):
        model_ckpt = os.path.join(model_weights_path, m, all_model_list[idx])
        VGG16_MODEL.load_weights(model_ckpt)
        # cnt = 100
        for case in testing_cases:
            print("processing %s " % case)
            if not os.path.exists(patch_root):
                print("Extracting patches")
                #TODO: add patch extraction

            eval_out_csv = os.path.join(output_dir, "eval_" + case + "_" + m + ".csv")
            csv_fp = open(eval_out_csv, 'w')
            # eval_results = "location_x,location_y,width,height,score0,score1,score2\n"
            eval_results = "location_x,location_y,width,height,Fibrosis,Cellularity,Orientation\n"

            f_list = glob.glob(os.path.join(patch_root, case, "*.jpg"))
            f_list = sorted(f_list)
            score_list = []
            location_list = []
            # ROI_location_list = []
            test_x = np.empty((1, img_rows, img_cols, img_channel), dtype=np.float32)
            # ROI_location = 0
            # ROI_width = 0
            # ROI_height = 0
            for f in f_list:
                test_x[0, :, :, :] = np.array(Image.open(f, 'r')).astype(np.float32) / 255.0
                out = VGG16_MODEL.predict(test_x)
                # get score for heat-map
                # max_val = max(out[0])
                # max_idx = list(out[0]).index(max_val)
                score = 0
                for val_idx, val in enumerate(out[0]):
                    score += val_idx * val
                score_list.append(score)

                # get locations from image file name
                img_fn = os.path.split(f)[1]
                ele = img_fn.split("_")
                loc_x = int(ele[-6]) + int(ele[-2])
                loc_y = int(ele[-5]) + int(ele[-1][0:-4])  # location should add top left coordinate
                # ROI_width = int(ele[-3])
                # ROI_height = int(ele[-4])
                # ROI_location = [int(ele[-6]), int(ele[-5])]
                # ROI_location_list.append([int(ele[-2]), int(ele[-1][0:-4])])
                location_list.append([loc_x, loc_y])
                eval_results += str(loc_x) + "," + str(loc_y) + "," + str(img_cols * img_rescale_rate) \
                                + "," + str(img_rows * img_rescale_rate) + "," + str(out[0][0]) + "," + str(out[0][1])\
                                + "," + str(out[0][2]) + "\n"
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
            # refer to this like: https://github.com/smujiang/Cytomine_GetAnnotation/blob/master/validate_exportation.py
            heat_map_img = np.zeros([thumb_img.height, thumb_img.width, 4]).astype(np.uint8)

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

            ## Create heatmap only for ROI
            # ROI_heat_map_img = np.zeros([ROI_height, ROI_width, 4]).astype(np.uint8)
            # for idx, loc in enumerate(ROI_location_list):
            #     start_x = int(loc[0])
            #     end_x = int((loc[0] + img_rows * img_rescale_rate))
            #     start_y = int(loc[1])
            #     end_y = int((loc[1] + img_cols * img_rescale_rate))
            #
            #     cmap_val = np.array(np.array(cmap(score_list[idx])) * 255).astype(np.uint8)
            #     x = np.broadcast_to(cmap_val, (end_y - start_y, end_x - start_x, 4))
            #     ROI_heat_map_img[start_y:end_y, start_x:end_x, :] = x
            # plt.imshow(ROI_heat_map_img)
            # plt.axis("off")
            # plt.show()
            # ROI_heat_map_img = Image.fromarray(ROI_heat_map_img)
            # if ROI_heat_map_img.mode == "RGB":
            #     a_channel = Image.new('L', ROI_heat_map_img.size, 255)  # 'L' 8-bit pixels, black and white
            #     ROI_heat_map_img.putalpha(a_channel)

            # Save
            if thumb_img.mode == "RGB":
                a_channel = Image.new('L', thumb_img.size, 255)  # 'L' 8-bit pixels, black and white
                thumb_img.putalpha(a_channel)
            blended = Image.blend(thumb_img, heat_map_img, 0.7)
            save_to = os.path.join(output_dir, case + "_" + m + "_blended.png")
            blended.save(save_to)
            save_to = os.path.join(output_dir, case + "_" + m + "_thumb.png")
            thumb_img.save(save_to)
            save_to = os.path.join(output_dir, case + "_" + m + "_heat_map.png")
            heat_map_img.save(save_to)
            # save_to = os.path.join(output_dir, case + "_" + m + "_ROI_heat_map.png")
            # ROI_heat_map_img.save(save_to)


