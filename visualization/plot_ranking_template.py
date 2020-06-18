import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import (Dict, List)
from statistics import mean

exp_id = 'temp'
# select_metric = 'Total Mag'

log_folder_dict = {  
                    Path(f"outputs/esst_1_r0.4_s0_ranking.csv"): "SST2",
                    Path(f"outputs/esubj_1_r0.4_s0_ranking.csv"): "SUBJ",
                    Path(f"outputs/etrec_1_r0.6_s0_ranking.csv"): "TREC",
                    }

def extract_top_ten_list(ranking_path):

    def average_list(l, n=5):
        l_avg = []
        for i in range(0, len(l), n):
            avg = mean(l[i:i+n])
            l_avg.append(avg)
        return l_avg

    flipped_ratio_csv_path = ranking_path
    lines = open(flipped_ratio_csv_path, 'r').readlines()
    top_ten_list = []
    for line in lines[1:]:
        parts = line[:-1].split(',')
        top_ten_list.append(int(parts[2]))
    
    return average_list(top_ten_list)

def plot_flipped_ratios(output_path, log_folder_dict):

    fig, ax = plt.subplots()

    for log_folder, name in log_folder_dict.items():
        top_ten_list_avg = extract_top_ten_list(log_folder)
        plt.plot( range(len(top_ten_list_avg)), top_ten_list_avg, label=f"{name}" )

    plt.legend(loc="upper right", prop={'size': 8})
    plt.xlabel("500 Minibatches")
    plt.ylabel("Number of noisy labels in the top 10 ranking examples")
    plt.savefig(output_path, dpi=400)
    print(output_path)

def extract_bottom_ten_list(ranking_path):

    def average_list(l, n=5):
        l_avg = []
        for i in range(0, len(l), n):
            avg = mean(l[i:i+n])
            l_avg.append(avg)
        return l_avg

    flipped_ratio_csv_path = ranking_path
    lines = open(flipped_ratio_csv_path, 'r').readlines()
    bottom_ten_list = []
    for line in lines[1:]:
        parts = line[:-1].split(',')
        bottom_ten_list.append(int(parts[3]))
    
    return average_list(bottom_ten_list)

def plot_flipped_ratios_bottom(output_path, log_folder_dict):

    fig, ax = plt.subplots()

    for log_folder, name in log_folder_dict.items():
        bottom_ten_list_avg = extract_bottom_ten_list(log_folder)
        plt.plot( range(len(bottom_ten_list_avg)), bottom_ten_list_avg, label=f"{name}" )

    plt.legend(loc="lower right", prop={'size': 8})
    plt.xlabel("500 Minibatches")
    plt.ylabel("Number of noisy labels in the bottom 10 ranking examples")
    plt.savefig(output_path, dpi=400)
    print(output_path)

if __name__ == "__main__":
    
    plot_flipped_ratios(f"plots/{exp_id}_flipped_ratio_combined_ten_top.png", log_folder_dict)
    plot_flipped_ratios_bottom(f"plots/{exp_id}_flipped_ratio_combined_ten_bottom.png", log_folder_dict)