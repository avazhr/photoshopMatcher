from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from arcface import ArcFace
import cv2
import sys 
import numpy as np 
import insightface 
from insightface.app import FaceAnalysis
# from insightface.data import get_image as ins_get_image
from PIL import Image
import os
import time
import random
import collections
import json

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
from sklearn import metrics

#from work_queue import *    # comment this out if developing locally or run:
                            # conda install -c conda-forge ndcctools

ORIG_PATH = "/Volumes/Extreme Pro/FRGC-2.0-dist/FRGC-2.0-dist/nd1/Fall2003"
RETOUCHED_PATH = "/Volumes/Extreme Pro/FacialRetouch"
FEATURES = ["eyes_100", "faceshape_100", "lips_100", "nose_100"]
DEST_DIRNAMES = {
    "eyes_50": "_eyes50",
    "eyes_100": "_eyes100",
    "faceshape_50": "_faceShape50",
    "faceshape_100": "_faceShape100",
    "lips_50": "_lips50",
    "lips_100": "_lips100",
    "nose_50": "_nose50",
    "nose_100": "_nose100"
}

app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))
face_rec = ArcFace.ArcFace()


def group_individuals(sorted_orig_dir, person_to_pic_range):

    cur_index = 0
    previd = ""
    for i, filename in enumerate(sorted_orig_dir):
        print(f"cur-index: {cur_index}, filename: {filename} ")
        if "d" not in filename or ".jpg" not in filename: 
            if previd != "":
                person_to_pic_range[cur_index].append(i)
                cur_index += 1
                previd = ""
            continue
        curid = filename.split("d")[0]
        if i == 0:
            person_to_pic_range.append([i])
        elif curid != previd:
            if previd != "":
                person_to_pic_range[cur_index].append(i)
                cur_index += 1
            person_to_pic_range.append([i])
            
        elif i == len(sorted_orig_dir) - 1:
            person_to_pic_range[cur_index].append(i)
        previd = curid

def get_embedding(path):

    try:
        img = cv2.imread(path)
        faces = app.get(img)
    except:
        return None

    face = faces[0].bbox.astype(np.int32)
    cropped = img[face[1]:face[3], face[0]:face[2]] 
    # cv2.imwrite("temp1.jpg", cropped1)
    # o1 = face_rec.calc_emb("temp1.jpg")
    emb = face_rec.calc_emb(cropped)

    return emb

def get_embedding_dist(emb1, emb2):
    if emb1 is None or emb2 is None:
        return None

    return face_rec.get_distance_embeddings(emb1, emb2)

def select(pool):
    res = random.sample(pool,1)
    pool.remove(res[0])
    return res[0]

def rand_pair(pool):

    return [select(pool) for i in range(2)]

def index_to_path(sorted_dir, path, index, feature):  
    
    if feature == "orig":
        return f"{path}/{sorted_dir[index]}"  
    else:
        person_id = sorted_dir[index].split(".")[0]
        print(f"{path}/{feature}/{person_id}{DEST_DIRNAMES[feature]}.jpg")
        return f"{path}/{feature}/{person_id}{DEST_DIRNAMES[feature]}.jpg"

   
if __name__ == "__main__":

    dist = {
        "same_person":{
            "orig":[],
            "eyes_100":[],
            "faceshape_100":[],
            "lips_100":[],
            "nose_100":[]
        },
        "imposter":{
            "orig":[],
            "eyes_100":[],
            "faceshape_100":[],
            "lips_100":[],
            "nose_100":[]
        }

    }
    
    orig_path = "/Volumes/Extreme Pro/FRGC-2.0-dist/FRGC-2.0-dist/nd1/Fall2003"
    retouched_path = "/Volumes/Extreme Pro/FacialRetouch"

    if len(sys.argv) == 3:
        orig_path = sys.argv[1]
        retouched_path = sys.argv[2]
    elif len(sys.argv) != 1:
        print('usage: ./arcfaceCompare.py <orig_path> <retouched_path>', file=sys.stderr)
        exit(1)
    
    sorted_orig_dir = sorted(os.listdir(orig_path))
    print(sorted_orig_dir)
    
    #marks the starting (inclusive) and ending (exclusive) index of the pictures of each person
    # person_to_pic_range = collections.defaultdict(lambda: list())
    person_to_pic_range = []
    group_individuals(sorted_orig_dir, person_to_pic_range)
    print(person_to_pic_range)
    
    people_pool = set([i for i in range(len(person_to_pic_range) )])
    print(people_pool)

    while len(people_pool) > 1 :
        #randomly pick a person
        ref_person_ind, imposter_ind = rand_pair(people_pool)
        ref_pic_ind, orig_pic_ind = rand_pair(person_to_pic_range[ref_person_ind])
        imposter_pic_ind = select(person_to_pic_range[imposter_ind])

        emb_ref = get_embedding( index_to_path(sorted_orig_dir, orig_path, ref_pic_ind, "orig") )
        # while emb_ref == None:
        #     ref_pic_ind = select(people_pool)

        # ref vs orig
        emb_orig = get_embedding( index_to_path(sorted_orig_dir, orig_path, orig_pic_ind, "orig") )
        # while emb_orig == None:
        #     orig_pic_ind = select(people_pool)
        orig_dist = get_embedding_dist(emb_ref, emb_orig)
        if orig_dist:
            dist["same_person"]["orig"].append(float(orig_dist))

        for feature in FEATURES:
            emb_orig_retouched = get_embedding( index_to_path(sorted_orig_dir, retouched_path, orig_pic_ind, feature) )
            same_feature_dist = get_embedding_dist(emb_ref, emb_orig_retouched)
            if same_feature_dist:
                dist["same_person"][feature].append(float(same_feature_dist))
            
        # ref vs imposter
        emb_imposter = get_embedding( index_to_path(sorted_orig_dir, orig_path, imposter_pic_ind, "orig") )

        impostor_dist = get_embedding_dist(emb_ref, emb_imposter)
        if impostor_dist:
            dist["imposter"]["orig"].append(float(impostor_dist))

        for feature in FEATURES:
            emb_imposter_retouched = get_embedding( index_to_path(sorted_orig_dir, retouched_path, imposter_pic_ind, feature) )
            impostor_feature_dist = get_embedding_dist(emb_ref, emb_imposter_retouched)
            if impostor_feature_dist:
                dist["imposter"][feature].append(float(impostor_feature_dist))

    # calculate the ROC curves

    # stores a dict of the ROC calculation results
    roc_results = {}

    # make a list of features that includes "orig"
    ROC_FEATURES = ["orig"]
    for f in FEATURES:
        ROC_FEATURES.append(f)

    for feature in ROC_FEATURES:
        # go feature by feature, including orig, and run each experiment
        gen_same = dist["same_person"][feature]
        gen_imp = dist["imposter"][feature]
        genuine_arr = gen_same + gen_imp
        y_true = [1] * len(gen_same) + [-1] * len(gen_imp)

        fpr, tpr, thresholds = metrics.roc_curve(np.array(y_true), np.array(genuine_arr))
        roc_results[feature] = (fpr.tolist(), tpr.tolist(), thresholds.tolist())
        print(" ")
        print(f"{feature}: {fpr},")
        print(f"{tpr},")
        print(f"{thresholds}")

        auc_score = metrics.roc_auc_score(np.array(y_true), np.array(genuine_arr))
        
        # plot curves
        plt.figure()
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC Curve, auc = {auc_score}"
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC for {feature}")
        plt.legend(loc="lower right")
        plt.savefig(f"{feature}.png")

    # output dist
    f = open("results.json", "w")
    json.dump(dist, f)
    f.close()

    # output ROC results
    f2 = open("roc_results.json", "w")
    json.dump(roc_results, f2)
    f2.close()
