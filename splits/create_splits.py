import pdb
import numpy as np
import random
import os

# randomly shuffle subjects
random.seed(30)
"""
!!! MOVED TO base experiment
Splits file_indices to train and val where subjects are unique to each set
"""
def group_by_subject(file_indices):     
    # Construct list of fp per subject name
    subject_fps = dict()  
    for fps in file_indices:
        # sub_idx = re.search(r'(?<=S_)\d*', fps[0]).group(0) # match by subject idx
        sub_idx = fps[0].split('-')[0]
        if sub_idx not in subject_fps:
            subject_fps[sub_idx] = fps
        else:
            subject_fps[sub_idx] = np.vstack([subject_fps[sub_idx], fps])
    
    # randomly shuffle subjects
    random.seed(30)
    subject_fps_items = list(subject_fps.items())
    random.shuffle(subject_fps_items)
    subject_fps = dict(subject_fps_items)
    return subject_fps   

def split(subject_fps, ratio=0.2):
    num_subjects = len(subject_fps)
    num_sub_per_split = int(num_subjects*ratio)
    agg_split_fps = []
    split_fps = None

    count = 0
    for i, (subject, fps) in enumerate(subject_fps.items()):
        # add first i subjects to train_subjects
        if count >= num_sub_per_split:
            agg_split_fps.append(split_fps)
            split_fps = None
            count = 0

        if count < num_sub_per_split:
            if split_fps is None:
                split_fps = fps
            else:
                split_fps = np.vstack([split_fps, fps])
            count = count + 1
    if split_fps is not None:
        agg_split_fps.append(split_fps)
    return agg_split_fps

dataset = "adni"
if dataset == "adni":
    save_dir = "cv/adni/"
    label1_fp = "img_Normal.npy"
    label2_fp = "img_AD.npy"
elif dataset == "aud":
    save_dir = "cv/aud/"
    label1_fp = "aud_img_C.npy"
    label2_fp = "aud_img_D.npy"
    
k = 5
ratio = 1.0 / k
# Load img_Normal.npy and img_AD.npy
fps_label1 = np.load(label1_fp)
fps_label2 = np.load(label2_fp)
# Combine to single list of pairs
fps = np.concatenate([fps_label1, fps_label2])
dataset_column = np.array([dataset]*len(fps)).reshape(-1, 1)
fps = np.hstack([fps, dataset_column])

# Group by subject 
subject_fps = group_by_subject(fps)

# Split into k folds
agg_split_fps = split(subject_fps, ratio=ratio)
# Save folds
for i in range(k):
    val_fold = agg_split_fps[i]
    train_folds = []
    for j in range(k):
        if i != j:
            train_folds.append(agg_split_fps[j])
    train_fold = np.concatenate(train_folds)
    # Save fold
    train_fold_fp = "{}_{}_train".format(dataset, i)
    val_fold_fp = "{}_{}_val".format(dataset, i)
    train_fold_fp = os.path.join(save_dir, train_fold_fp)
    val_fold_fp = os.path.join(save_dir, val_fold_fp)
    np.save(train_fold_fp, train_fold)
    np.save(val_fold_fp, val_fold)
    print("Saved fold: {}".format(i))
pdb.set_trace()
