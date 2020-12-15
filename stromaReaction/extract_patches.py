from wsitools.patch_extraction.patch_extractor import ExtractorParameters, PatchExtractor
from wsitools.tissue_detection.tissue_detector import TissueDetector  # import dependent packages
import os

output_root = "/infodev1/non-phi-data/junjiang/OvaryCancer/StromaReaction/WSIs_patches_256"
WSI_root = "/infodev1/non-phi-data/junjiang/OvaryCancer/StromaReaction/WSIs"
log_dir = "/infodev1/non-phi-data/junjiang/OvaryCancer/StromaReaction/WSIs_patch_extraction_log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# testing_cases = ["Wo-2-F5_RIO1338_HE", "Wo-2-L1_RIO1338_HE", "Wo-2-N1_RIO1338_HE",
#                  "Wo-3-B5_RIO1338_HE", "Wo-3-C1_RIO1338_HE", "Wo-4-A9_RIO1338_HE",
#                  "Wo-4-B1_RIO1338_HE"]
testing_cases = ["Wo-1-A5_RIO1338_HE", "Wo-1-C4_RIO1338_HE", "Wo-1-F1_RIO1338_HE", "Wo-2-B4_RIO1338_HE", "Wo-2-F1_RIO1338_HE"]

wsi_ext = ".svs"


tissue_detector = TissueDetector("LAB_Threshold", threshold=85)
parameters = ExtractorParameters(output_root, log_dir=log_dir, save_format='.jpg', patch_size=512, stride=512, sample_cnt=-1, extract_layer=0, patch_filter_by_area=0.5, patch_rescale_to=256)
patch_extractor = PatchExtractor(tissue_detector, parameters, feature_map=None, annotations=None)

for case in testing_cases:
    print("Extracting from %s" % case + wsi_ext)
    wsi_fn = os.path.join(WSI_root, case + wsi_ext)
    patch_num = patch_extractor.extract(wsi_fn)
    output_dir = os.path.join(output_root, case)
    print("%d Patches have been save to %s" % (patch_num, output_dir))

