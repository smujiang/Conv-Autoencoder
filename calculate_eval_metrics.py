import os
import numpy as np

borderline_case_id_list = ["OCMC-016", "OCMC-017", "OCMC-018", "OCMC-019", "OCMC-020",
                           "OCMC-021", "OCMC-022", "OCMC-023", "OCMC-024", "OCMC-025",
                           "OCMC-026", "OCMC-027", "OCMC-028", "OCMC-029", "OCMC-030"]
high_grade_case_id_list = ["OCMC-001", "OCMC-002", "OCMC-003", "OCMC-004", "OCMC-005",
                           "OCMC-006", "OCMC-007", "OCMC-008", "OCMC-009", "OCMC-010",
                           "OCMC-011", "OCMC-012", "OCMC-013", "OCMC-014", "OCMC-015"]

data_root_dir = "/infodev1/non-phi-data/junjiang/OvaryCancer/PatchAEClassification/reconstructed"

output_summary = os.path.join(data_root_dir, "summary.csv")
fp = open(os.path.join(output_summary), 'w')
wrt_str = "case,err_mean,err_std\n"
all_cases = borderline_case_id_list + high_grade_case_id_list
for c in all_cases:
    fn = os.path.join(data_root_dir, c + ".csv")
    errors = []
    lines = open(fn, 'r').readlines()
    for l in lines[1:]:
        errors.append(float(l.split(",")[2]))
    # err = np.array(errors)
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    wrt_str += c + "," + str(mean_err) + "," + str(std_err) + "\n"

fp.write(wrt_str)
fp.close()









