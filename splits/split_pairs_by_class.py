import pdb
import os
import numpy as np
import pandas as pd
"""
Splits pairs of images from img1.txt and img2.txt based on classes for ADNI dataset
Saves images_{class1}.txt and images_{class2}.txt where each file contains pairs
"""
def preprocess_adni_splits():
    # load pairs
    img1_c_fp = os.path.expanduser("~/loca/splits/img1_c.txt")
    img2_c_fp = os.path.expanduser("~/loca/splits/img2_c.txt")        
    file_idx1 = np.genfromtxt(img1_c_fp, dtype='str') 
    file_idx2 = np.genfromtxt(img2_c_fp, dtype='str')
    file_indices = np.stack([file_idx1, file_idx2], axis=1)

    # load labels
    label_fp = os.path.expanduser("~/data/adni/ADNI1_subjects_with_img.csv")
    labels_df = pd.read_csv(label_fp)

    # split file_indices to different classes
    fp_by_classes = {label : [] for label in labels_df["dx_label"].unique()}
    
    for i, indices in enumerate(file_indices):
        fp1 = indices[0]
        subject_name = fp1.split("-")[0]
        dx = labels_df.loc[labels_df["subjects"]==subject_name]["dx_label"].to_numpy()[0]
        fp_by_classes[dx].append([indices[0], indices[1]])
    print("Done splitting fp_by_classes")

    split_dir = os.path.expanduser("~/loca/splits")
    for label in labels_df["dx_label"].unique():
        img_label_fp = os.path.join(split_dir, "img_{}".format(label))
        images_by_label = np.asarray(fp_by_classes[label])
        np.save(img_label_fp, images_by_label)
        print("Saved: {}".format(img_label_fp))
    print("Finished preprocessing splits")

"""
Splits pairs of images from img1.txt and img2.txt based on classes for AUD dataset
Saves images_{class1}.txt and images_{class2}.txt where each file contains pairs
"""
def preprocess_aud_splits():
    # load pairs
    img1_c_fp = os.path.expanduser("~/loca/splits/aud_img1.txt")
    img2_c_fp = os.path.expanduser("~/loca/splits/aud_img2.txt")        
    file_idx1 = np.genfromtxt(img1_c_fp, dtype='str') 
    file_idx2 = np.genfromtxt(img2_c_fp, dtype='str')
    file_indices = np.stack([file_idx1, file_idx2], axis=1)

    # load labels
    label_fp = os.path.expanduser("~/data/lab_data/demographics_lab.csv")
    labels_df = pd.read_csv(label_fp)
    # split file_indices to different classes
    fp_by_classes = {label : [] for label in labels_df["demo_diag"].unique()}
    
    for i, indices in enumerate(file_indices):
        fp1 = indices[0]
        subject_name = fp1.split("-")[0]
        dx = labels_df.loc[labels_df["subject"]==subject_name]["demo_diag"].to_numpy()[0]
        fp_by_classes[dx].append([indices[0], indices[1]])
    print("Done splitting fp_by_classes")

    split_dir = os.path.expanduser("~/loca/splits")
    for label in labels_df["demo_diag"].unique():
        img_label_fp = os.path.join(split_dir, "aud_img_{}".format(label))
        images_by_label = np.asarray(fp_by_classes[label])
        np.save(img_label_fp, images_by_label)
        print("Saved: {}".format(img_label_fp))
    print("Finished preprocessing splits")

if __name__ == '__main__':
    preprocess_adni_splits()
    preprocess_aud_splits()