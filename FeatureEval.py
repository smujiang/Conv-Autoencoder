import os
import numpy as np
import matplotlib.pyplot as plt

patch_root = "/infodev1/non-phi-data/junjiang/OvaryCancer/auto_enc_patches_256"
data_train_txt = os.path.join(patch_root, "training_shuffled.txt")

# load labels
lines = open(data_train_txt, 'r').readlines()
labels = []
for l in lines:
    labels.append(int(l.split(",")[0]))

# load features
data_file = "/infodev1/non-phi-data/junjiang/OvaryCancer/PatchAEClassification/PatchEmbeddings/training_data_part_80348.npz"
temp_data = np.load(data_file)["arr_0"]
patch_features_data = temp_data.reshape([temp_data.shape[0], -1])
patch_labels = np.array(labels[0:patch_features_data.shape[0]])

# dump data to file
# csv_dump = "/infodev1/non-phi-data/junjiang/OvaryCancer/PatchAEClassification/PatchEmbeddings/csv_dump.csv"
# wrt_str = ""
# p_cnt = 1000
# n_cnt = 1000
# for idx, p in enumerate(patch_labels):
#     pf = patch_features_data[idx, :]
#     pf_str = ','.join(['%.5f' % num for num in pf])
#     if p == 0:
#         if n_cnt > 0:
#             wrt_str += str(p) + "," + pf_str + "\n"
#             n_cnt -= 1
#     if p == 1:
#         if p_cnt > 0:
#             wrt_str += str(p) + "," + pf_str + "\n"
#             p_cnt -= 1
#     if p_cnt == 0 and n_cnt == 0:
#         break
#
# csv_fp = open(csv_dump, 'w')
# csv_fp.write(wrt_str)
# csv_fp.close()
#

'''
# Feature importance
'''
# from sklearn import linear_model
# clf = linear_model.Lasso(alpha=1e-4, max_iter=1000, normalize=True)     # normalize = False
# clf.fit(patch_features_data, np.array(labels[0:patch_features_data.shape[0]]))
# importance = clf.coef_
# # all zeros?
# plt.figure(figsize=[12, 8])
# plt.plot(range(len(importance)), importance)
#
# k = len(set(importance))   # 65  / 8192
# km = []
# for imp in importance:
#     if not abs(imp) == 0:
#         km.append(imp)
#
# mk = len(km)  # 64
# plt.show()

'''Coefficient'''
from scipy import stats
all_per_coef = np.zeros([2, 2])
all_sp_coef = np.zeros([2, 2])
per_calculatable = 0
sp_calculatable = 0
for j in range(patch_features_data.shape[1]):
    jth_per_coef = np.corrcoef(patch_features_data[:, j], patch_labels)
    jth_sp_coef = stats.spearmanr(patch_features_data[:, j], patch_labels)
    if np.isnan(jth_per_coef).any():
        f = patch_features_data[:, j]
        l = patch_labels
        print("detect nan")
    else:
        per_calculatable += 1
        all_per_coef += jth_per_coef

    if np.isnan(jth_sp_coef).any():
        # f = patch_features_data[:, j]
        # l = patch_labels
        print("detect nan")
    else:
        sp_calculatable += 1
        all_sp_coef += jth_sp_coef
    #
    # plt.hist(jth_coef.flatten(), density=True, bins=30)  # `density=False` would make counts
    # plt.ylabel('Probability')
    # plt.xlabel('coefficients')
    # plt.title("All Pearson coefficients distribution")
    # plt.show()

print(all_per_coef/per_calculatable)
print(all_sp_coef/sp_calculatable)














# pos_patch_feature_data = patch_features_data[patch_labels == 1]
# neg_patch_feature_data = patch_features_data[patch_labels == 0]
#
# pos_cnt = pos_patch_feature_data.shape[0]
# neg_cnt = neg_patch_feature_data.shape[0]
# if pos_cnt > neg_cnt:
#     pos_patch_feature_data = pos_patch_feature_data[0:neg_cnt, :]
# if pos_cnt < neg_cnt:
#     neg_patch_feature_data = neg_patch_feature_data[0:pos_cnt, :]
#
#
#
# '''
# # Pearson product-moment correlation coefficients
# # Spearman rank-correlation coefficients
# '''
#
# all_coef = np.corrcoef(pos_patch_feature_data, neg_patch_feature_data)
# plt.hist(all_coef.flatten(), density=True, bins=30)  # `density=False` would make counts
# plt.ylabel('Probability')
# plt.xlabel('coefficients')
# plt.title("All Pearson coefficients distribution")
# plt.show()
# all_p_max = max(all_coef.flatten())
# all_p_min = min(all_coef.flatten())
#
# all_col_coef = np.corrcoef(pos_patch_feature_data, neg_patch_feature_data, rowvar=False)
# plt.hist(all_col_coef.flatten(), density=True, bins=30)  # `density=False` would make counts
# plt.ylabel('Probability')
# plt.xlabel('coefficients')
# plt.title("All Pearson coefficients distribution")
# plt.show()
# all_cp_max = max(all_col_coef.flatten())
# all_cp_min = min(all_col_coef.flatten())
#
# from scipy import stats
#
# pe_coef_list = []
# sp_coef_list = []
# sp_pval_list = []
# for j in range(pos_patch_feature_data.shape[1]):
#     pe_coef = np.corrcoef(pos_patch_feature_data[:, j], neg_patch_feature_data[:, j])
#     sp_coef = stats.spearmanr(pos_patch_feature_data[:, j], neg_patch_feature_data[:, j])
#     pe_coef_list.append(pe_coef[0, 1])
#     sp_coef_list.append(sp_coef.correlation)
#     sp_pval_list.append(sp_coef.pvalue)
#
# plt.hist(pe_coef_list, density=True, bins=30)  # `density=False` would make counts
# plt.ylabel('Probability')
# plt.xlabel('coefficients')
# plt.title("Pearson coefficients distribution")
# plt.show()
# pe_max = max(pe_coef_list)
# pe_min = min(pe_coef_list)
#
#
# plt.hist(sp_coef_list, density=True, bins=30)  # `density=False` would make counts
# plt.ylabel('Probability')
# plt.xlabel('coefficients')
# plt.title("Spearman coefficients distribution")
# plt.show()
# sp_max = max(sp_coef_list)
# sp_min = min(sp_coef_list)
#
#
# print("Done")
#
#
#










