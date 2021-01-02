import numpy as np
import pandas as pd

samples_csv = "/infodev1/non-phi-data/junjiang/OvaryCancer/StromaReaction/PatchSampling/all_samples.csv"
anno_df = pd.read_csv(samples_csv, header=0)

Fibrosis_scores_list = pd.Series.get(anno_df, "Fibrosis").tolist()
Cellularity_scores_list = pd.Series.get(anno_df, "Cellularity").tolist()
Orientation_scores_list = pd.Series.get(anno_df, "Orientation").tolist()



F_C_r = np.corrcoef(Fibrosis_scores_list, Cellularity_scores_list)
O_C_r = np.corrcoef(Orientation_scores_list, Cellularity_scores_list)
F_O_r = np.corrcoef(Orientation_scores_list, Fibrosis_scores_list)
all_r = np.corrcoef([Fibrosis_scores_list, Orientation_scores_list, Orientation_scores_list])

print(F_C_r)
print(O_C_r)
print(F_O_r)










