import os, cv2
import numpy as np

save_files = 'save_hist'

folders = os.listdir(save_files)
stages = 4

for folder in folders:
    for stage in range(stages):
        after_path = os.path.join(save_files, folder, "block{}_mask_afterDrop.png".format(stage+1))
        before_path = os.path.join(save_files, folder, "block{}_mask_beforeDrop.png".format(stage+1))
        after_img = cv2.imread(after_path)
        before_img = cv2.imread(before_path)
        bound = np.full((after_img.shape[0], int(after_img.shape[1]/12), 3), 0, dtype="uint8")
        merged_im = np.concatenate((before_img, bound, after_img), axis=1)
        cv2.imshow("stage{}".format(stage+1), merged_im)
    cv2.waitKey(0)

