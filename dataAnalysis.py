import numpy as np
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from sklearn import metrics
import matplotlib.colors as mcolors
import pandas as pd
import scipy.stats as stats
import seaborn as sns

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

# 1. plot the distributions between genuine, retouched, and imposter
def plot_dist(data):

    for feature in data["same_person"]:
        if feature == "orig": continue
        print(feature)
        gen, gen_retouched, imp = data["same_person"]["orig"], data["same_person"][feature], data["imposter"]["orig"]
        binwidth = 0.1
        plt.figure()
        plt.hist(gen, bins=np.arange(min(gen), max(gen) + binwidth, binwidth), alpha=0.5, label="genuine")
        plt.hist(gen_retouched, bins=np.arange(min(gen_retouched), max(gen_retouched) + binwidth, binwidth), alpha=0.5, label=f"{feature}_retouched")
        plt.hist(imp, bins=np.arange(min(imp), max(imp) + binwidth, binwidth), alpha=0.5, label="impostor")
        plt.legend(loc='upper right')
        plt.gca().set(title=f"Distribution for {feature}", ylabel='Frequency')
        plt.savefig(f"results/{feature}_imp_dist.png")
        

# 2. d prime
def d_prime(gen_scores, imp_scores):
    return np.sqrt(2)*abs(np.mean(gen_scores)-np.mean(imp_scores))

def calc_d_prime(data):

    res = []

    for category in data:
        for feature in data[category]:
            res.append(d_prime(data["same_person"]["orig"], data[category][feature]))

    return res

def plot_d_prime(data):

    columns = [f"orig vs. {b}" if a== "same_person" else f"imp vs. {b}" for a in data for b in data[a]]
    print(columns)
    d_prime = calc_d_prime(data)
    print(d_prime)  

    fig, ax = plt.subplots(figsize =(16, 9))
    ax.barh(columns, d_prime)
    
    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)
    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad = 5)
    ax.yaxis.set_tick_params(pad = 10)
    
    # Add x, y gridlines
    ax.grid(b = True, color ='grey',
            linestyle ='-.', linewidth = 0.5,
            alpha = 0.2)
    # Show top values
    ax.invert_yaxis()
    
    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width()+0.2, i.get_y()+0.5,
                str(round((i.get_width()), 5)),
                fontsize = 10, #fontweight ='bold',
                color ='grey')
    
    # Add Plot Title
    ax.set_title('D prime value for different comparisons', fontsize = 20,
                loc ='left', )

    
    # Show Plot
    #plt.show()
    plt.savefig(f"results/dprime_bar.png")   

# 3. roc curves
def plot_all_roc(data):

    roc_results = {}

    i = 0
    plt.figure()

    for feature in data["same_person"]:  
        if '50' in feature: continue

        gen_same = data["same_person"][feature]
        gen_imp = data["imposter"][feature]
        genuine_arr = [*gen_same, *gen_imp] # concatenate gen_same and gen_imp
        y_true = [-1] * len(gen_same)
        y_true.extend([1] * len(gen_imp))

        fpr, tpr, thresholds = metrics.roc_curve(np.array(y_true), np.array(genuine_arr))
        print(f"fpr:{fpr}")
        print(f"tpr:{tpr}")
        #raise KeyError
        fpr_trimmed = np.where(fpr > 0.0000000001, fpr, -10)
        tpr_trimmed = np.where(tpr > 0.0000000001, tpr, -10)
        tpr_trimmed_logged = np.log10(fpr_trimmed, out=fpr_trimmed, where=fpr_trimmed > 0)
        print(f"tpr_trimmed_logged:{tpr_trimmed_logged}")

        plt.plot(
            np.log10(fpr_trimmed, out=fpr_trimmed, where=fpr_trimmed > 0),
            np.log10(tpr_trimmed, out=tpr_trimmed, where=tpr_trimmed > 0),
            color= list(mcolors.TABLEAU_COLORS)[i],
            lw=2,
            label=f"ROC Curve, feature = {feature}"
        )
        i += 1

    # plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([-10.0, 1.0])
    plt.ylim([-3.0, 1.0])
    plt.xlabel("Log_10 of False Positive Rate")
    plt.ylabel("Log_10 of True Positive Rate")
    plt.title(f"ROC")
    plt.legend(loc="lower right")
    plt.savefig("results/roc.png")            

# 4. anova 
def anova():
    df = pd.read_json("results.json")
    # json_struct = json.loads(df.to_json(orient="results.json"))
    same_df = df["same_person"]
    same_df.index = [
        "orig_same", 
        "eyes_100_same", 
        "faceshape_100_same", 
        "lips_100_same", 
        "nose_100_same",
        "eyes_50_same",
        "faceshape_50_same",
        "lips_50_same",
        "nose_50_same"
    ] # finish this and do the same for impostor
    imp_df = df["imposter"]
    imp_df.index = [
        "orig_imp",
        "eyes_100_imp",
        "faceshape_100_imp",
        "lips_100_imp",
        "nose_100_imp",
        "eyes_50_imp",
        "faceshape_50_imp",
        "lips_50_imp",
        "nose_50_imp"
    ]
    df_flat = pd.concat([same_df, imp_df])
    
    
    df_flat.columns = ["experiment", "values"]
    print(df_flat.to_string())

    EXPERIMENTS = ["eyes", "faceshape", "lips", "nose"]
    ORIG_SAME = "orig_same"
    ORIG_IMP = "orig_imp"
    for e in EXPERIMENTS:
        # run same trial
        same_50 = f"{e}_50_same"
        same_100 = f"{e}_100_same"
        # 50
        fvalue, pvalue = stats.ttest_ind(df_flat[ORIG_SAME], df_flat[same_50], equal_var=True)
        if pvalue < 0.05:
            print(f"H0 rejected for {ORIG_SAME}, {same_50}")
            print(fvalue, pvalue)
        # 100
        fvalue, pvalue = stats.ttest_ind(df_flat[ORIG_SAME], df_flat[same_100], equal_var=True)
        if pvalue < 0.05:
            print(f"H0 rejected for {ORIG_SAME}, {same_100}")
            print(fvalue, pvalue)

        # run impostor trial
        imp_50 = f"{e}_50_imp"
        imp_100 = f"{e}_100_imp"
        # 50
        fvalue, pvalue = stats.ttest_ind(df_flat[ORIG_IMP], df_flat[imp_50], equal_var=True)
        if pvalue < 0.05:
            print(f"H0 rejected for {ORIG_IMP}, {imp_50}")
            print(fvalue, pvalue)        
        # 100
        fvalue, pvalue = stats.ttest_ind(df_flat[ORIG_IMP], df_flat[imp_100], equal_var=True)
        if pvalue < 0.05:
            print(f"H0 rejected for {ORIG_IMP}, {imp_100}")
            print(fvalue, pvalue)

        # run 100 same vs 100 imp trial
        # fvalue, pvalue = stats.f_oneway(df_flat[ORIG_SAME], df_flat[same_100], df_flat[imp_100])
        # if pvalue > 0.05:
        #     print(f"H0 rejected for {ORIG_SAME}, {same_100}, {imp_100}")
        #     print(fvalue, pvalue)

# 5. equal error rate
def calc_eer(data):

    res = []

    for category in data:
        for feature in data[category]:
            
            gen_same = data["same_person"][feature]
            gen_imp = data["imposter"][feature]
            genuine_arr = [*gen_same, *gen_imp] # concatenate gen_same and gen_imp
            y_true = [-1] * len(gen_same)
            y_true.extend([1] * len(gen_imp))

            fpr, tpr, thresholds = metrics.roc_curve(np.array(y_true), np.array(genuine_arr), pos_label=1)

            eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            thresh = interp1d(fpr, thresholds)(eer)

            res.append(eer)
        break

    return res

def plot_eer(data):

    # columns = [f"orig vs. {b}" if a== "same_person" else f"imp vs. {b}" for a in data for b in data[a]]
    columns = [f"orig vs. {a}" for a in data["same_person"]]
    print(columns)
    d_prime = calc_eer(data)
    print(d_prime)  

    fig, ax = plt.subplots(figsize =(16, 9))
    ax.barh(columns, d_prime)
    
    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)
    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad = 5)
    ax.yaxis.set_tick_params(pad = 10)
    
    # Add x, y gridlines
    ax.grid(b = True, color ='grey',
            linestyle ='-.', linewidth = 0.5,
            alpha = 0.2)
    # Show top values
    ax.invert_yaxis()
    
    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width(), i.get_y()+0.5,
                str(round((i.get_width()), 5)),
                fontsize = 10, fontweight ='bold',
                color ='grey')
    
    # Add Plot Title
    ax.set_title('EER for different comparisons', fontsize = 20,
                loc ='left', )

    
    # Show Plot
    #plt.show()
    plt.savefig(f"results/eer_bar.png")   


if __name__ == "__main__":

    f = open("results.json")

    data = json.load(f)

    # plot_d_prime(data)
    # plot_dist(data)
    # plot_all_roc(data)

    anova()
    # print(plot_eer(data))
