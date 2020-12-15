import numpy as np
import os
import pandas as pd

patch_root = "/infodev1/non-phi-data/junjiang/OvaryCancer/StromaReaction/PatchSampling"
testing_result_dir = "/infodev1/non-phi-data/junjiang/OvaryCancer/StromaReaction/VGG16_Classification_ROIs"
eval_out_dir = "/infodev1/non-phi-data/junjiang/OvaryCancer/StromaReaction/eval"

all_class_score_txt = ["score0","score1","score2"]
all_class_list = ["Fibrosis", "Cellularity", "Orientation"]
testing_cases = ["Wo-1-A5_RIO1338_HE", "Wo-1-C4_RIO1338_HE", "Wo-1-F1_RIO1338_HE", "Wo-2-B4_RIO1338_HE",
                 "Wo-2-F1_RIO1338_HE"]

# annotation_csv = "/infodev1/non-phi-data/junjiang/OvaryCancer/StromaReaction/PatchSampling/all_samples.csv"
annotation_csv = "/infodev1/non-phi-data/junjiang/OvaryCancer/StromaReaction/PatchSampling/validation_five_cases.csv"


anno_df = pd.read_csv(annotation_csv, header=0)
file_list = pd.Series.get(anno_df, "img_fn").tolist()
Fibrosis_scores_list = pd.Series.get(anno_df, "Fibrosis").tolist()
Cellularity_scores_list = pd.Series.get(anno_df, "Cellularity").tolist()
Orientation_scores_list = pd.Series.get(anno_df, "Orientation").tolist()

# save_all_testing_score_list = os.path.join(eval_out_dir, "all_testing_scores.npy")
save_all_testing_score_list = os.path.join(eval_out_dir, "validation_scores.npy")

if os.path.exists(save_all_testing_score_list):
    all_testing_score_list = np.load(save_all_testing_score_list)
else:
    all_testing_score_list = []

    testing_df_list = []
    for case in testing_cases:  # case ID
        case_df_list = []
        for class_txt in all_class_list:  # metric ID
            csv_name = "eval" + "_" + case + "_" + class_txt + ".csv"
            csv_full_name = os.path.join(testing_result_dir, csv_name)
            testing_df = pd.read_csv(csv_full_name, header=0)
            case_df_list.append(testing_df)
        testing_df_list.append(case_df_list)

    for img_fn in file_list:
        ele = os.path.split(img_fn)[1].split("_")
        patch_loc_x = int(ele[-2])
        patch_loc_y = int(ele[-1][0:-4])
        roi_loc_x = int(ele[-6])
        roi_loc_y = int(ele[-5])

        case_id = ele[0] + "_" + ele[1] + "_" + ele[2]

        testing_scores_list = []
        for c_idx, class_txt in enumerate(all_class_list):
            df = testing_df_list[testing_cases.index(case_id)][c_idx]
            testing_patch_loc_x_list = pd.Series.get(df, "location_x").tolist()
            testing_patch_loc_y_list = pd.Series.get(df, "location_y").tolist()
            testing_patch_scores = pd.Series.get(df, all_class_score_txt)

            Found = False
            for idx, loc_x in enumerate(testing_patch_loc_x_list):
                loc_y = testing_patch_loc_y_list[idx]
                if (loc_x == roi_loc_x + patch_loc_x) and (loc_y == roi_loc_y + patch_loc_y):
                    # print("match")
                    Found = True

                    scores = testing_patch_scores.iloc[idx].tolist()
                    testing_scores_list.append(scores.index(max(scores)))
                    break
            if not Found:
                raise Exception("not found")
        #
        # testing_df_list = []
        # for class_txt in all_class_list:
        #     csv_name = "eval" + "_" + ele[0] + "_" + ele[1] + "_" + ele[2] + "_" + class_txt + ".csv"
        #     csv_full_name = os.path.join(testing_result_dir, csv_name)
        #     testing_df = pd.read_csv(csv_full_name, header=0)
        #     testing_df_list.append(testing_df)
        #
        # testing_scores_list = []
        # for c_idx, df in enumerate(testing_df_list):
        #     testing_patch_loc_x_list = pd.Series.get(df, "location_x").tolist()
        #     testing_patch_loc_y_list = pd.Series.get(df, "location_y").tolist()
        #     testing_patch_scores = pd.Series.get(df, all_class_score_txt[c_idx]).tolist()
        #     Found = False
        #     for idx, loc_x in enumerate(testing_patch_loc_x_list):
        #         loc_y = testing_patch_loc_y_list[idx]
        #         if (loc_x == roi_loc_x + patch_loc_x) and (loc_y == roi_loc_y + patch_loc_y):
        #             # print("match")
        #             Found = True
        #             testing_scores_list.append(testing_patch_scores[idx])
        #             break
        #     if not Found:
        #         raise Exception("not found")

        all_testing_score_list.append(testing_scores_list)
    np.save(save_all_testing_score_list, np.array(all_testing_score_list))

# calculate confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn
import matplotlib.pyplot as plt

def plot_confusion_matrix(data, labels, title, output_filename):
    """Plot confusion matrix using heatmap.

    Args:
        data (list of list): List of lists with confusion matrix data.
        labels (list): Labels which will be plotted across x and y axis.
        output_filename (str): Path to output file.

    """
    seaborn.set(color_codes=True)
    plt.figure(1, figsize=(9, 6))

    plt.title(title)

    seaborn.set(font_scale=1.4)
    ax = seaborn.heatmap(data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'})

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set(ylabel="True Label", xlabel="Predicted Label")

    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close()

labels = ["0", "1", "2"]
anno_score_list = [Fibrosis_scores_list, Cellularity_scores_list, Orientation_scores_list]
for idx, y_true in enumerate(anno_score_list):
    print(all_class_list[idx])
    y_pred = all_testing_score_list[:, idx]
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    title = all_class_list[idx] + " Confusion Matrix"
    output_filename = os.path.join(eval_out_dir, all_class_list[idx] + ".jpg")
    plot_confusion_matrix(cm, labels, title, output_filename)

    cm = confusion_matrix(y_true, y_pred, normalize='true')
    title = all_class_list[idx] + " Normalized Confusion Matrix"
    output_filename = os.path.join(eval_out_dir, all_class_list[idx] + "_normalized.jpg")
    plot_confusion_matrix(cm, labels, title, output_filename)
    print(cm)

print("OK")



















