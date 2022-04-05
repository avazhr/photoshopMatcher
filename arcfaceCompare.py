from inspect import getmembers
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
# FEATURES = ["eyes_100", "faceshape_100", "lips_100", "nose_100"]
FEATURES = ["eyes_100", "faceshape_100", "lips_100", "nose_100", "eyes_50", "faceshape_50", "lips_50", "nose_50"]
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
        #print(f"cur-index: {cur_index}, filename: {filename} ")
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
    try:
        emb = face_rec.calc_emb(cropped)
    except:
        print(f"Error calculating embedding for {path}")
        return None

    return emb

def get_embedding_dist(emb1, emb2):
    if emb1 is None or emb2 is None:
        return None

    return face_rec.get_distance_embeddings(emb1, emb2)

def select(pool, people_pool, ref_person_ind):
    res = random.sample(pool,1)
    pool.remove(res[0])
    if len(pool) == 0 or len(pool) == 1:
        people_pool.remove(ref_person_ind)
    return res[0]

def rand_pair(pool, people_pool, ref_person_ind):

    return [select(pool, people_pool, ref_person_ind) for i in range(2)]

def rand_people_pair(people_pool):
    return random.sample(people_pool,2)

def index_to_path(individual_id, photo_number, path, feature):  
    
    if feature == "orig":
        print(f"{path}/{individual_id}d{photo_number}.jpg")
        return f"{path}/{individual_id}d{photo_number}.jpg"  
    else:
        print(f"{path}/{feature}/{individual_id}d{photo_number}{DEST_DIRNAMES[feature]}.jpg")
        return f"{path}/{feature}/{individual_id}d{photo_number}{DEST_DIRNAMES[feature]}.jpg"

def plot_graph(dist):

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
        genuine_arr = [*gen_same, *gen_imp]    # check concatenation
        print(f"genuine_arr: {genuine_arr}")
        # are these the right values for these parts of y_true?
        y_true = [-1] * len(gen_same)
        y_true.extend([1] * len(gen_imp))
        print(f"y_true: {y_true}")

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

        # plot ROC distributions
        # if both are Gaussian/completely overlap, we likely have a problem with ArcFace
        # comparisons
        plt.figure()
        plt.hist(gen_same, bins=30, alpha=0.5, label="genuine")
        plt.hist(gen_imp, bins=30, alpha=0.5, label="impostor")
        plt.legend(loc='upper right')
        plt.gca().set(title=f"Distribution for {feature}", ylabel='Frequency')
        plt.savefig(f"{feature}_dist.png")

    # output dist
    f = open("results.json", "w")
    json.dump(dist, f)
    f.close()

    # output ROC results
    f2 = open("roc_results.json", "w")
    json.dump(roc_results, f2)
    f2.close()

    return 

class Individual:

    # init will automatically find all of this individual's photos and put them in 
    # the object's photos array
    def __init__(self, id):
        self.id = id
        self.photos = []
        d = os.scandir(ORIG_PATH)
        for entry in d:
            if entry.is_file() and id == entry.name.split('d')[0]:
                # get the second half of the filename that identifies this photo
                # and chop off the .jpg extension
                photo = entry.name.split('d')[1]
                self.photos.append(photo[:-4])

        print(f"{self.id}: {self.photos}")
        # shuffle the photos array so every picture is randomized
        random.shuffle(self.photos)

    def next_photo(self):
        return self.photos.pop()
   
if __name__ == "__main__":

    dist = {
        "same_person":{
            "orig":[],
            "eyes_100":[],
            "faceshape_100":[],
            "lips_100":[],
            "nose_100":[],
            "eyes_50": [],
            "faceshape_50": [],
            "lips_50": [],
            "nose_50": []
        },
        "imposter":{
            "orig":[],
            "eyes_100":[],
            "faceshape_100":[],
            "lips_100":[],
            "nose_100":[],
            "eyes_50": [],
            "faceshape_50": [],
            "lips_50": [],
            "nose_50": []
        }

    }
    
    orig_path = "/Volumes/Extreme Pro/FRGC-2.0-dist/FRGC-2.0-dist/nd1/Fall2003"
    retouched_path = "/Volumes/Extreme Pro/FacialRetouch"

    if len(sys.argv) == 3:
        orig_path = sys.argv[1]
        retouched_path = sys.argv[2]
    elif len(sys.argv) != 1:
        print('usage: python3 arcfaceCompare.py <orig_path> <retouched_path>', file=sys.stderr)
        exit(1)

    # find all the individuals in the data set and make objects for them
    people = []
    people_seen = set()

    # people.append(Individual("02463"))
    # people.append(Individual("04202"))

    d = os.scandir(ORIG_PATH)
    for entry in d:
        if entry.is_file():
            # extract the individual's ID
            id = entry.name.split('d')[0]
            if id in people_seen:
                continue

            else:
                # add this individual's object to the people array
                people_seen.add(id)
                people.append(Individual(id))

    d.close()
    # randomize the array so new comparisons are being made each time
    random.shuffle(people)

    num_people = len(people)

    # error file for debugging
    error_file = open("error.txt", "w")

    # get all the distances
    while people:
        # boolean lets us make self vs. self comparisons even if there isn't an impostor
        impostor_exists = True
        # randomly pick two people, ref and impostor
        # these two people will be used in the below loops to iterate over all their photos
        ref_person = people.pop()
        impostor_person = None
        if people:
            impostor_person = people.pop()
        else:
            impostor_exists = False

        print(" ")
        print(f"{len(people)} / {num_people} individuals remaining")
        print(" ")

        # MARK: process ref_person
        while ref_person.photos:

            print(ref_person.photos)

            # randomly select two genuine pictures from the same individual
            ref_pic = ref_person.next_photo()   # this will be a string
            # ref embedding will be used for all comparisons
            emb_ref = get_embedding( index_to_path(ref_person.id, ref_pic, orig_path, "orig") )

            # make sure there is another original photo to make a comparison
            if ref_person.photos:
                compare_pic = ref_person.next_photo()
                emb_compare = get_embedding( index_to_path(ref_person.id, compare_pic, orig_path, "orig") )
                orig_dist = get_embedding_dist(emb_ref, emb_compare)

                if orig_dist:
                    dist["same_person"]["orig"].append(float(orig_dist))
                else:
                    if emb_ref is None:
                        error_file.write(f"{ref_person.id}d{ref_pic}")
                    if emb_compare is None:
                        error_file.write(f"{ref_person.id}d{compare_pic}")

            # MARK: repeat the process with a random photo from the impostor
            # we use index 0 because the array has been randomized
            if impostor_exists:
                # note that we don't pop impostor photos; we're going to check ref photo
                # against all the impostor photos
                imp_photo = impostor_person.photos[0]
                emb_impostor = get_embedding(index_to_path(impostor_person.id, imp_photo, orig_path, "orig"))
                impostor_dist = get_embedding_dist(emb_ref, emb_impostor)
                if impostor_dist:
                    dist["imposter"]["orig"].append(float(impostor_dist))

            # MARK: now move on to the feature comparisons
            for feature in FEATURES:
                # for same person
                emb_orig_retouched = get_embedding(index_to_path(ref_person.id, ref_pic, retouched_path, feature))
                orig_retouched_dist = get_embedding_dist(emb_ref, emb_orig_retouched)
                if orig_retouched_dist:
                    dist["same_person"][feature].append(float(orig_retouched_dist))

                # MARK: for impostor. using random impostor photo as with the unmodified experiment above
                if impostor_exists:
                    imp_photo = impostor_person.photos[0]
                    emb_impostor_retouched = get_embedding(index_to_path(impostor_person.id, imp_photo, retouched_path, feature))
                    imp_retouched_dist = get_embedding_dist(emb_ref, emb_impostor_retouched)
                    if imp_retouched_dist:
                        dist["imposter"][feature].append(float(imp_retouched_dist))

    plot_graph(dist)
    error_file.close()