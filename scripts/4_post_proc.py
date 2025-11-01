#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import glob
from tqdm import tqdm

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, BinaryClassifierOutputTarget
import cv2
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Function
import torchvision
from torchvision import transforms
from torchvision import models
from torchvision.models.resnet import Bottleneck
from torchvision.models import resnet50, efficientnet_v2_l, MobileNetV2
from glob import glob
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import json
from scipy import stats
from random import shuffle
import random

#%%

def average_grad_cam():
    # Normal: 0, Accident: 1
    random_seed = 42
    gradcam_dir = "./results/xai_results/gradcam"
    meta_dir = "./results/xai_results/gradcam_metadata"

    model_names = ["resnext", "efficient", "mobile", "VGG", "resnet"]
    dists = [5, 10, 20, 30]

    for model_name in model_names:
        for dist in dists:
            gradcam_files = glob(f"{gradcam_dir}/{model_name}_{dist}/*.parquet")
            # print(f"{meta_dir}/metadata_{model_name}_{dist}.json")
            with open(f"{meta_dir}/metadata_{model_name}_{dist}.json", "r") as f:
                meta_file = json.load(f)

            norm_as_norm_grads_whole, acc_as_acc_grads_whole = np.zeros((224, 224)), np.zeros((224, 224))
            norm_as_norm_count, acc_as_acc_count = 0, 0
            for file_name, info in tqdm(meta_file.items()):
                df = pd.read_parquet(f"{gradcam_dir}/{model_name}_{dist}/{file_name}")
                if info["label"] == 0 and info["pred"] == 0:
                    norm_as_norm_grads_whole += df.values
                    norm_as_norm_count += 1
                elif info["label"] == 1 and info["pred"] == 1:
                    acc_as_acc_grads_whole += df.values
                    acc_as_acc_count += 1
                else: continue
            acc_as_acc_grads_whole = acc_as_acc_grads_whole / acc_as_acc_count
            norm_as_norm_grads_whole = norm_as_norm_grads_whole / norm_as_norm_count
            
            # print(acc_as_acc_grads_whole.flatten().max(), norm_as_norm_grads_whole.flatten().max())
            fig, axs = plt.subplots(1, 2)
            fig.set_figheight(5)
            fig.set_figwidth(10)
            axs[0].imshow(acc_as_acc_grads_whole / acc_as_acc_count)
            axs[0].set_title("Accident to Accident")
            axs[1].imshow(norm_as_norm_grads_whole / norm_as_norm_count)
            axs[1].set_title("Normal to Normal")
            plt.suptitle(f"{model_name}_{dist} - Accuracy: {(acc_as_acc_count + norm_as_norm_count) / len(meta_file) * 100:.2f}%")
            plt.tight_layout()
            plt.savefig(f"./results/imgs/{random_seed}/average_gradcam_{model_name}_{dist}.png")
            plt.close()

            # break
        # break

# average_grad_cam()

#%%
def get_average_rtr():
    model_name = "VGG"
    dists = [10, 20, 30]
    random_seed = 123
    meta_dir = f"./results/xai_results/gradcam_metadata/{random_seed}/0_001/1"
    for r_idx, dist in enumerate(dists):
        print(model_name, dist)

        average_normal_rtr, average_accident_rtr = np.zeros((dist*20+1, dist*20+1)), np.zeros((dist*20+1, dist*20+1))
        with open(f"{meta_dir}/metadata_{model_name}_{dist}.json", "r") as f:
            meta_file = json.load(f)

        for path in tqdm(glob(os.path.join(f"./data/exp_data/123/{dist}/test", "**", "*.parquet"), recursive=True)):
            file_name = os.path.basename(path)
            
            info = meta_file[file_name]

            df = pd.read_parquet(path)
            # print(df.values.shape)
            if info["label"] == 0 and info["pred"] == 0:
                average_normal_rtr += df.values
            elif info["label"] == 1 and info["pred"] == 1:
                average_accident_rtr += df.values

        fig = plt.figure(frameon=False, figsize=(12,12))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(np.log1p(average_normal_rtr))
        plt.axis("off")
        plt.grid(False)
        plt.savefig(os.path.join("./results/imgs", f"Log1p_Normal_Average_RTR_{dist}"))
        plt.close()
        # plt.show()

        fig = plt.figure(frameon=False, figsize=(12,12))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(np.log1p(average_accident_rtr))
        plt.axis("off")
        plt.grid(False)
        plt.savefig(os.path.join("./results/imgs", f"Log1p_Accident_Average_RTR_{dist}"))
        plt.close()
        # plt.show()
#%%

from scipy import stats
from random import shuffle
import random

def get_stat_results(model_name, method, fix_seed=123):

    random.seed(fix_seed)
    
    def get_center_grid(center_idx, center_length, esd):
        
        center_r_idxs = np.arange(center_idx - center_length/2, center_idx + center_length/2)
        center_c_idxs = np.arange(center_idx - center_length/2, center_idx + center_length/2)
        center_r_idxs, center_c_idxs = np.meshgrid(center_r_idxs, center_c_idxs)
        center_r_idxs, center_c_idxs = center_r_idxs.flatten().astype(int), center_c_idxs.flatten().astype(int)
        center_grid = esd[center_r_idxs, center_c_idxs].reshape(center_length, center_length)
        anchor = np.mean(center_grid)
        
        return center_grid, anchor

    def generate_crust_mask(esd, left_corner, step_size, anchor):
        mask = np.zeros_like(esd, dtype=bool)
                        
        left_up = left_corner
        right_down = len(esd) - left_up
        exterior_r_idxs = np.arange(left_up, right_down)
        exterior_c_idxs = np.arange(left_up, right_down)
        exterior_r_idxs, exterior_c_idxs = np.meshgrid(exterior_r_idxs, exterior_c_idxs)
        exterior_r_idxs, exterior_c_idxs = exterior_r_idxs.flatten().astype(int), exterior_c_idxs.flatten().astype(int)
        mask[exterior_r_idxs, exterior_c_idxs] = True
                        
        left_up = left_corner + step_size
        right_down = len(esd) - left_up
        interior_r_idxs = np.arange(left_up, right_down)
        interior_c_idxs = np.arange(left_up, right_down)
        interior_r_idxs, interior_c_idxs = np.meshgrid(interior_r_idxs, interior_c_idxs)
        interior_r_idxs, interior_c_idxs = interior_r_idxs.flatten().astype(int), interior_c_idxs.flatten().astype(int)
        mask[interior_r_idxs, interior_c_idxs] = False
        
        return mask

    def calc_dist(esd, center_idx, center_length, step_size):
        center_grid, anchor = get_center_grid(center_idx, center_length, esd)
        variances = []
        variances_grid = np.zeros((224,224))
        for left_corner in range(0, int(center_idx), step_size):
            mask = generate_crust_mask(esd, left_corner, step_size, anchor)
                            
            crust_vals = esd[mask]
            variance = np.mean((crust_vals - anchor)**2)
            variances_grid[mask] = variance
            variances.append(variance)
                            
        variances = np.array(variances)
        gradient_variances = np.gradient(variances)
        deltas = variances[:-1] - variances[1:]
        del_deltas = abs(deltas[:-1] - deltas[1:])
                        
        x_axis = list(range(0, int(center_idx), step_size))
        max_idx = np.argmax(del_deltas) +1 # Index is shifted by one, because it is calculated by the difference between two points
        distance = center_idx - x_axis[max_idx]
        
        return distance, variances_grid, variances, gradient_variances

    center_idx = 112
    center_length = 8
    step_size = 8
    random_seed = 123
    # method = "gradcam" # "gradcam", "shap", "lime"

    gradcam_dir = f"./results/xai_results/{method}/{random_seed}/0_001/1"
    meta_dir = f"./results/xai_results/{method}_metadata/{random_seed}/0_001/1"

    model_name = model_name
    dists = [10, 20, 30]

    results = []

    for r_idx, dist in enumerate(dists):
        print(model_name, dist)

        average_normal_rtr, average_accident_rtr = np.zeros((dist*20+1, dist*20+1)), np.zeros((dist*20+1, dist*20+1))
        with open(f"{meta_dir}/metadata_{model_name}_{dist}.json", "r") as f:
            meta_file = json.load(f)
        
        # Valute init
        x_axis = list(range(0, int(center_idx), step_size))
        TTs_dist, FFs_dist, TFs_dist, FTs_dist = [], [], [], []
        norm_as_norm_grads_whole, acc_as_acc_grads_whole = np.zeros((224, 224)), np.zeros((224, 224))
        norm_as_norm_count, acc_as_acc_count = 0, 0
        TF_count, FT_count = 0, 0

        # Shuffle
        area_set = set()
        keys =  list(meta_file.keys())
        if dist == 30: random.seed(280138)
        shuffle(keys)
        meta_file = {key: meta_file[key] for key in keys}


        for idx, (file_name, info) in enumerate(tqdm((meta_file.items()))):
            "2019_07_03_7_0_35.0475_129.0106"
            year, month, day, hour, minute, lat, lon = file_name.rstrip(".parquet").split("_")
            unique_key = f"{year}_{month}_{day}_{hour}_{lat}_{lon}"
            if unique_key in area_set: continue

            df = pd.read_parquet(f"{gradcam_dir}/{model_name}_{dist}/{file_name}")
            distance, _, _, _ = calc_dist(df.values, center_idx, center_length, step_size)

            if info["label"] == 0 and info["pred"] == 0:
                norm_as_norm_grads_whole += df.values
                norm_as_norm_count += 1
                if unique_key not in area_set:
                    TTs_dist.append(distance)
                    results.append([file_name, dist, "Low" if info["pred"] == 0 else "High", distance])
            elif info["label"] == 1 and info["pred"] == 1:
                acc_as_acc_grads_whole += df.values
                acc_as_acc_count += 1
                if unique_key not in area_set:
                    FFs_dist.append(distance)
                    results.append([file_name, dist, "Low" if info["pred"] == 0 else "High", distance])
            elif info["label"] == 0 and info["pred"] == 1:
                TF_count += 1
            elif info["label"] == 1 and info["pred"] == 0:
                FT_count += 1

            area_set.add(unique_key)


    results = pd.DataFrame(results, columns=["file_name", "radii", "risk", "distance"])
    results.to_csv(f"./results/csvs/threshold_shap.csv", index=False)


def print_out_stat_results():
    min_val = float("inf")
    min_rand = 0
    for fix_seed in [7866]:
        "7866 (8, 8), 280138(8, 8, 30)"
        results = get_stat_results(model_name="VGG", fix_seed=int(fix_seed))

        dist10_true, dist20_true, dist30_true = True, True, False
        p_vals = []
        for dist in results["radii"].unique():
            test_stat, p_val = stats.kstest(results.loc[(results["radii"] == dist) & (results["risk"] == "Low"), "distance"],\
                                            results.loc[(results["radii"] == dist) & (results["risk"] == "High"), "distance"])
            p_vals.append(p_val)
            print(f"Dist {dist} kstest - statistics: {test_stat}/ p-value: {p_val}")
            test_stat, p_val = stats.ks_2samp(results.loc[(results["radii"] == dist) & (results["risk"] == "Low"), "distance"],\
                                            results.loc[(results["radii"] == dist) & (results["risk"] == "High"), "distance"])
            p_vals.append(p_val)

            print(f"Dist {dist} ks2samp - statistics: {test_stat}/ p-value: {p_val}")
            test_stat, p_val = stats.ttest_ind(results.loc[(results["radii"] == dist) & (results["risk"] == "Low"), "distance"],\
                                            results.loc[(results["radii"] == dist) & (results["risk"] == "High"), "distance"])
            p_vals.append(p_val)

            print(f"Dist {dist} t-test - statistics: {test_stat}/ p-value: {p_val}")

        if dist10_true and dist20_true and dist30_true:
            print("\n\n", fix_seed, p_vals, "\n\n")
# print_out_stat_results()

def get_point_variance(model_name, fix_seed=123, point_idx=0):

    random.seed(fix_seed)
    
    def get_center_grid(center_idx, center_length, esd):
        
        center_r_idxs = np.arange(center_idx - center_length/2, center_idx + center_length/2)
        center_c_idxs = np.arange(center_idx - center_length/2, center_idx + center_length/2)
        center_r_idxs, center_c_idxs = np.meshgrid(center_r_idxs, center_c_idxs)
        center_r_idxs, center_c_idxs = center_r_idxs.flatten().astype(int), center_c_idxs.flatten().astype(int)
        center_grid = esd[center_r_idxs, center_c_idxs].reshape(center_length, center_length)
        anchor = np.mean(center_grid)
        
        return center_grid, anchor

    def generate_crust_mask(esd, left_corner, step_size, anchor):
        mask = np.zeros_like(esd, dtype=bool)
                        
        left_up = left_corner
        right_down = len(esd) - left_up
        exterior_r_idxs = np.arange(left_up, right_down)
        exterior_c_idxs = np.arange(left_up, right_down)
        exterior_r_idxs, exterior_c_idxs = np.meshgrid(exterior_r_idxs, exterior_c_idxs)
        exterior_r_idxs, exterior_c_idxs = exterior_r_idxs.flatten().astype(int), exterior_c_idxs.flatten().astype(int)
        mask[exterior_r_idxs, exterior_c_idxs] = True
                        
        left_up = left_corner + step_size
        right_down = len(esd) - left_up
        interior_r_idxs = np.arange(left_up, right_down)
        interior_c_idxs = np.arange(left_up, right_down)
        interior_r_idxs, interior_c_idxs = np.meshgrid(interior_r_idxs, interior_c_idxs)
        interior_r_idxs, interior_c_idxs = interior_r_idxs.flatten().astype(int), interior_c_idxs.flatten().astype(int)
        mask[interior_r_idxs, interior_c_idxs] = False
        
        return mask

    def calc_kval(esd, center_idx, center_length, step_size, point_idx):
        center_grid, anchor = get_center_grid(center_idx, center_length, esd)
        # print("center_grid", center_grid)
        # print("anchor", anchor)
        variances = []
        variances_grid = np.zeros((224,224))

        mask = generate_crust_mask(esd, point_idx, step_size, anchor)
        crust_vals = esd[mask]
        variance = np.mean((crust_vals - anchor)**2)
        variances_grid[mask] = variance
        variances.append(variance)

        mask = generate_crust_mask(esd, point_idx+step_size, step_size, anchor)
        crust_vals = esd[mask]
        variance = np.mean((crust_vals - anchor)**2)
        variances_grid[mask] = variance
        variances.append(variance)
                            
        variances = np.array(variances)
        gradient_variances = np.gradient(variances)
                        
        return gradient_variances[0]


    center_idx = 112
    center_length = 8
    step_size = 8
    random_seed = 123
    method = "gradcam"

    gradcam_dir = f"./results/xai_results/{method}/{random_seed}/0_001/1"
    meta_dir = f"./results/xai_results/{method}_metadata/{random_seed}/0_001/1"

    model_name = model_name
    dists = [10, 20, 30]

    results = []

    lands = [
        "2019_07_03_7_35.0475_129.0106",
        "2019_07_03_11_35.0475_129.0106",
        "2019_07_29_7_35.0650_129.0114",
        "2019_07_29_11_35.0650_129.0114",
        "2019_08_03_13_35.5194_129.3761",
        "2019_08_03_17_35.5194_129.3761",
        "2019_11_07_6_35.6517_129.4514",
        "2019_11_07_10_35.6517_129.4514",
        "2019_12_12_6_35.4772_129.4292",
        "2019_12_12_10_35.4772_129.4292",
        "2021_01_15_2_35.0939_129.0294",
        "2021_01_15_6_35.0939_129.0294",
        "2021_02_24_11_35.5233_129.3736",
        "2021_02_24_15_35.5233_129.3736",
        "2021_02_24_12_35.4833_129.4167",
        "2021_02_24_16_35.4833_129.4167",
        "2021_03_08_6_40_35.1044_128.9311", 
        "2021_03_08_10_40_35.1044_128.9311",
        "2021_03_23_9_5_34.6056_127.7239",
        "2021_03_23_13_5_34.6056_127.7239",
        "2021_05_14_10_34.0267_127.3075",
        "2021_05_14_14_34.0267_127.3075",
        "2021_06_12_7_34.6833_125.4333",
        "2021_06_12_11_34.6833_125.4333",
        "2021_06_12_13_34.4667_127.4425",
        "2021_06_12_17_34.4667_127.4425"
        "2021_06_13_22_34.7333_127.7333",
        "2021_06_14_2_34.7333_127.7333",

    ]
    seas = [
        "2019_07_04_17_35.4000_129.4333",
        "2019_08_08_12_35.0417_129.0278",
        "2019_08_08_16_35.0417_129.0278",
        "2019_09_02_2_35.5500_129.4667",
        "2019_09_02_6_35.5500_129.4667",
        "2019_11_07_10_55_34.9667_128.9667",
        "2019_11_07_14_55_34.9667_128.9667",
        "2021_01_31_12_0_35.1031_129.1558",
        "2021_01_31_16_0_35.1031_129.1558",
        "2021_02_03_17_20_34.4769_127.9872",
        "2021_02_03_21_20_34.4769_127.9872",
        "2021_02_10_2_35.1383_129.1664",
        "2021_02_10_6_35.1383_129.1664",
        "2021_02_25_4_34.3167_128.0500",
        "2021_02_25_8_34.3167_128.0500",
        "2021_04_06_1_35.1075_129.4100",
        "2021_04_17_9_34.5053_128.2714",
        "2021_04_17_13_34.5053_128.2714",
        "2021_05_20_6_35.0189_129.0211",
        "2021_05_20_10_35.0189_129.0211",
        "2021_05_27_17_34.3417_128.3950",
        "2021_05_27_21_34.3417_128.3950",
        "2021_06_02_23_35.0064_129.3844",
        "2021_06_03_3_35.0064_129.3844",
        "2021_06_22_3_34.8083_128.3683",
        "2021_06_22_7_34.8083_128.3683",
        "2021_06_27_12_35.4167_129.5667",
        "2021_06_27_16_35.4167_129.5667",
        "2021_06_29_10_35.0694_129.0281",
        "2021_06_29_14_35.0694_129.0281",
    ]

    for r_idx, dist in enumerate(dists):
        print(model_name, dist)

        average_normal_rtr, average_accident_rtr = np.zeros((dist*20+1, dist*20+1)), np.zeros((dist*20+1, dist*20+1))
        with open(f"{meta_dir}/metadata_{model_name}_{dist}.json", "r") as f:
            meta_file = json.load(f)
        
        
        
        x_axis = list(range(0, int(center_idx), step_size))
        TTs_dist, FFs_dist, TFs_dist, FTs_dist = [], [], [], []
        norm_as_norm_grads_whole, acc_as_acc_grads_whole = np.zeros((224, 224)), np.zeros((224, 224))
        norm_as_norm_count, acc_as_acc_count = 0, 0
        TF_count, FT_count = 0, 0
        # rand_idx = np.random.randint(0, len(meta_file)-1)
        area_set = set()
        keys =  list(meta_file.keys())
        shuffle(keys)
        meta_file = {key: meta_file[key] for key in keys}
        for idx, (file_name, info) in enumerate(tqdm((meta_file.items()))):
            year, month, day, hour, minute, lat, lon = file_name.rstrip(".parquet").split("_")
            unique_key = f"{year}_{month}_{day}_{hour}_{lat}_{lon}"
            if unique_key in area_set: continue
            if unique_key in seas: place="seas"
            else: place="land"

            df = pd.read_parquet(f"{gradcam_dir}/{model_name}_{dist}/{file_name}")
            k_val = calc_kval(df.values, center_idx, center_length, step_size, point_idx)

            if info["label"] == 0 and info["pred"] == 0:
                norm_as_norm_grads_whole += df.values
                norm_as_norm_count += 1
                if unique_key not in area_set:
                    TTs_dist.append(k_val)
            elif info["label"] == 1 and info["pred"] == 1:
                acc_as_acc_grads_whole += df.values
                acc_as_acc_count += 1
                if unique_key not in area_set:
                    FFs_dist.append(k_val)
            elif info["label"] == 0 and info["pred"] == 1:
                TF_count += 1
            elif info["label"] == 1 and info["pred"] == 0:
                FT_count += 1

            area_set.add(unique_key)
            results.append([unique_key, dist, info["pred"], place, k_val])
    
    results = pd.DataFrame(results, columns=["key", "dist", "risk", "place", "k_val"])

    return results

#%%

def plot_point_variance(model_name, fix_seed, point_idx):
    min_val = float("inf")
    min_rand = 0
    p_vals = []
    for fix_seed in [7866]:
        "7866 (8, 8), 280138(8, 8, 30)"
        
        for point_idx in range(0, 104, 8):
            print(point_idx)
            results = get_point_variance(model_name="VGG", fix_seed=int(fix_seed), point_idx = point_idx)

            for dist in results["dist"].unique():
                for place in results["place"].unique():
                    test_stat, p_val = stats.kstest(results.loc[(results["dist"] == dist) & (results["risk"] == 0) & (results["place"] == place), "k_val"], \
                                                    results.loc[(results["dist"] == dist) & (results["risk"] == 1) & (results["place"] == place), "k_val"])
                    p_vals.append([dist, point_idx, place, p_val])

    p_vals = pd.DataFrame(p_vals, columns=["dist", "k_val", "place", "p_val"])

    for dist in p_vals["dist"].unique():
        for place in p_vals["place"].unique():
            df = p_vals.loc[(p_vals["dist"] == dist) & (p_vals["place"] == place), ["p_val", "k_val"]]
            df.set_index("k_val", inplace=True)

            plt.plot(df, linewidth=3)
            plt.hlines(0.05, 0, 96, colors="red", linestyles="dashed", linewidth=2)
            plt.plot([], c="red", linestyle="dashed", label="p=0.05")
            plt.xticks(np.linspace(0, 96, 5), [f'{i:.2f}km' for i in np.linspace(0, dist, 5)][::-1])
            plt.xlabel("K-values", fontsize=16)
            plt.ylabel("p-value", fontsize=16)
            plt.legend(loc="upper center", fontsize=16)
            # plt.savefig(f"p_vals_{dist}.png", dpi=300)
            plt.savefig("./results/imgs/p_vals_{place}_{dist}.png", dpi=300)
            plt.title([], fontsize=1)
            plt.tight_layout()
            plt.close()
                


#%%

def draw_boxplot(results):
    fig, axs = plt.subplots(1, 3)
    fig.set_figheight(6)
    fig.set_figwidth(14)

    c="red"; c2 = "blue"; back_c = "white"
    for idx, dist in enumerate(results.keys()):
        axs[idx].boxplot(results[dist]["accident"], positions=[0.25], notch=True, patch_artist=True,
                    boxprops=dict(facecolor=back_c, color=c),
                    # capprops=dict(color=c),
                    whiskerprops=dict(color=c),
                    flierprops=dict(color=c, markeredgecolor=c),
                    medianprops=dict(color="black"),
                    )
        axs[idx].boxplot(results[dist]["normal"], positions=[-0.25], notch=True, patch_artist=True,
                    boxprops=dict(facecolor=back_c, color=c2),
                    capprops=dict(color=c2),
                    whiskerprops=dict(color=c2),
                    flierprops=dict(color=c2, markeredgecolor=c2),
                    medianprops=dict(color="black"),
                    )
        axs[idx].set_title(f"{dist}km")
        axs[idx].set_yticks(np.linspace(0, 112, 5))
        axs[idx].set_yticklabels(np.linspace(0, dist, 5))
        axs[idx].set_ylim(-5, 112)
        axs[idx].set_xticks([])

        if idx == 1:
            axs[idx].plot([], label="High risk", color=c, linewidth=3)
            axs[idx].plot([], label="Low risk", color=c2, linewidth=3)
            axs[idx].legend(loc="lower left", fontsize=12)

    plt.show()
#%%
