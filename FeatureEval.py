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
'''
# Feature importance
'''
from sklearn import linear_model
clf = linear_model.Lasso(alpha=1e-4)
clf.fit(patch_features_data, np.array(labels[0:patch_features_data.shape[0]]))
importance = clf.coef_
# all zeros?


''''''
pos_patch_feature_data = patch_features_data[patch_labels == 1]
neg_patch_feature_data = patch_features_data[patch_labels == 0]

pos_cnt = pos_patch_feature_data.shape[0]
neg_cnt = neg_patch_feature_data.shape[0]
if pos_cnt > neg_cnt:
    pos_patch_feature_data = pos_patch_feature_data[0:neg_cnt, :]
if pos_cnt < neg_cnt:
    neg_patch_feature_data = neg_patch_feature_data[0:pos_cnt, :]



'''
# Pearson product-moment correlation coefficients
# Spearman rank-correlation coefficients 
'''

all_coef = np.corrcoef(pos_patch_feature_data, neg_patch_feature_data)
plt.hist(all_coef.flatten(), density=True, bins=30)  # `density=False` would make counts
plt.ylabel('Probability')
plt.xlabel('coefficients')
plt.title("All Pearson coefficients distribution")
plt.show()
all_p_max = max(all_coef.flatten())
all_p_min = min(all_coef.flatten())

all_col_coef = np.corrcoef(pos_patch_feature_data, neg_patch_feature_data, rowvar=False)
plt.hist(all_col_coef.flatten(), density=True, bins=30)  # `density=False` would make counts
plt.ylabel('Probability')
plt.xlabel('coefficients')
plt.title("All Pearson coefficients distribution")
plt.show()
all_cp_max = max(all_col_coef.flatten())
all_cp_min = min(all_col_coef.flatten())

from scipy import stats

pe_coef_list = []
sp_coef_list = []
sp_pval_list = []
for j in range(pos_patch_feature_data.shape[1]):
    pe_coef = np.corrcoef(pos_patch_feature_data[:, j], neg_patch_feature_data[:, j])
    sp_coef = stats.spearmanr(pos_patch_feature_data[:, j], neg_patch_feature_data[:, j])
    pe_coef_list.append(pe_coef[0, 1])
    sp_coef_list.append(sp_coef.correlation)
    sp_pval_list.append(sp_coef.pvalue)

plt.hist(pe_coef_list, density=True, bins=30)  # `density=False` would make counts
plt.ylabel('Probability')
plt.xlabel('coefficients')
plt.title("Pearson coefficients distribution")
plt.show()
pe_max = max(pe_coef_list)
pe_min = min(pe_coef_list)


plt.hist(sp_coef_list, density=True, bins=30)  # `density=False` would make counts
plt.ylabel('Probability')
plt.xlabel('coefficients')
plt.title("Spearman coefficients distribution")
plt.show()
sp_max = max(sp_coef_list)
sp_min = min(sp_coef_list)


print("Done")













