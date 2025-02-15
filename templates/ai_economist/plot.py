import json
import os
import os.path as osp
import pickle
import pandas as pd
import glob
import re
import PIL
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict

###############################################################################
# LOAD FINAL JSON & NPY RESULTSS
###############################################################################

folders = os.listdir("./")
final_results = {}
train_info = {}

for folder in folders:
    if folder.startswith("run") and osp.isdir(folder):
        with open(osp.join(folder, "final_info.json"), "r") as f:
            final_results[folder] = json.load(f)

        all_data = np.load(osp.join(folder, "training_logs/all_results.npy"), allow_pickle=True).item()
        train_info[folder] = all_data
        print(all_data.keys())

###############################################################################
# EXTRACT USEFUL INFO
###############################################################################
# Initialize plot data
all_train_keys = all_data.keys() #['per_agent_episode_return', 'mean_population_episode_return', 'government_reward', 'total_episode_return_sum', 'total_loss', 'actor_loss', 'value_loss', 'entropy', 'training timestep', 'productivity', 'equality', 'per_agent_coin', 'per_agent_labor', 'mean_population_coin', 'mean_population_labor', 'median_population_coin', 'median_population_labor', 'median_population_episode_return']
plot_data = {}

for folder, data in final_results.items():
    #sub_key = list(data.keys())[0]
    #sub_dict = data[sub_key]
    #metrics = {k: v["means"] for k, v in sub_dict.items()}
    sub_key = list(data.keys())[0]
    sub_dict = data[sub_key]
    metrics = sub_dict["means"] 
    
    actor_loss = metrics.get("actor_loss")
    equality = metrics.get("equality")
    productivity = metrics.get("productivity")
    total_loss = metrics.get("total_loss")
    timestep = metrics.get("training timestep", 0)  # fallback if missing

    plot_data[folder] = {
        "actor_loss": actor_loss,
        "total_loss": total_loss,
        "equality": equality,
        "productivity": productivity,
        "training_timestep": timestep
    }

    
###############################################################################
# CREAT LEGEND and COLOR PALETTE
###############################################################################

# CREATE LEGEND -- PLEASE FILL IN YOUR RUN NAMES HERE
# Keep the names short, as these will be in the legend.
labels = {
    "run_0": "Baselines",
}



# Create a programmatic color palette
def generate_color_palette(n):
    cmap = plt.get_cmap('tab20') # You can change 'tab20' to other colormaps like 'Set1', 'Set2', 'Set3', etc.
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n)]

# Use the run key as the default label if not specified
runs = list(final_results.keys())
colors = generate_color_palette(len(runs))
run_labels = labels

# CREATE TRAINING PLOTS
###############################################################################
# PLOT 1: ACTOR_LOSS VS. ITERATIONS
###############################################################################

plt.figure(figsize=(8, 6))
for i, run in enumerate(runs):
    data = plot_data[run]
    actor_loss = data.get("actor_loss")
    timestep   = data.get("training_timestep")
    if actor_loss is None:
        continue
    plt.scatter([timestep], [actor_loss], label=run_labels[run], color=colors[i])

plt.title("Actor Loss over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Actor Loss")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("plot_actor_loss.png")
plt.close()
###############################################################################
# PLOT 2: EQUALITY VS. ITERATIONS
###############################################################################
plt.figure(figsize=(8, 6))
for i, run in enumerate(runs):
    data = plot_data[run]
    eq = data.get("equality")
    ts = data.get("training_timestep")
    if eq is None:
        continue
    plt.scatter([ts], [eq], label=run_labels[run], color=colors[i])

plt.title("Equality over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Equality")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("plot_equality.png")
plt.close()



###############################################################################
# PLOT 3: SCATTER PLOT OF EQUALITY VS. PRODUCTIVITY
###############################################################################
plt.figure(figsize=(6, 6))
for i, run in enumerate(runs):
    data = plot_data[run]
    eq = data.get("equality")
    prod = data.get("productivity")
    if eq is None or prod is None:
        continue
    plt.scatter(eq, prod, label=run_labels[run], color=colors[i])

plt.title("Equality vs. Productivity")
plt.xlabel("Equality")
plt.ylabel("Productivity")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("scatter_eq_vs_prod.png")
plt.close()

###############################################################################
# PLOT EVALUATION GRAPHS
###############################################################################

full_key_list = ['equality', 'inventory_coin', 'inventory_labor', 'population_actions', 'population_utility', 'productivity', 'skills_craft', 'tax_rates', 'timestep', 'trade_price_history'] #feel free to use more of these to create additional graphs, some values are dictionaries!

def plot_all_evaluations(out_dir):
    eval_dir = os.path.join(out_dir, "eval_logs")
    eval_files = glob.glob(os.path.join(eval_dir, "evaluation_logs_*.pkl"))
    
    if not eval_files:
        print("No evaluation files found.")
        return

    runs = []

    for file_path in eval_files:
        match = re.search(r'evaluation_logs_(\d+)\.pkl', file_path)
        if not match:
            print(f"Skipping non-standard file: {file_path}")
            continue

        run_id = match.group(1)
        
        try:
            with open(file_path, 'rb') as f:
                data_list = pickle.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

        if not isinstance(data_list, list):
            print(f"Unexpected data format in {file_path}, skipping.")
            continue

        data_dict = {
            key: [entry.get(key) for entry in data_list if isinstance(entry, dict)]
            for key in ['equality', 'productivity', 'timestep']
        }

        # if data_list and isinstance(data_list[-1], dict):
        #     print("Last element keys:", list(data_list[-1].keys()))
        runs.append({'eval_id': run_id, **data_dict})

    if not runs:
        print("No valid runs to plot.")
        return
    plot_info = [
        ('equality',    'Equality Comparison Across Runs'),
        ('productivity', 'Productivity Comparison Across Runs'),
        ('timestep',    'Timestep Progression Validation')
    ]

    plt.figure(figsize=(12, 12))
    for i, (data_key, plot_title) in enumerate(plot_info, start=1):
        plt.subplot(len(plot_info), 1, i)
        for run_data in runs:
            plt.plot(run_data['timestep'], run_data[data_key], label=f"Eval {run_data['eval_id']}")
        plt.title(plot_title)
        plt.xlabel('Timestep')
        plt.ylabel(data_key.capitalize())
        plt.grid(True)
        plt.legend()
    plt.tight_layout()
    folder_name = os.path.basename(out_dir.rstrip(os.sep))
    out_file = f"evaluation_plots_{folder_name}.png"
    plt.savefig(out_file)
    plt.close()


def plot_run_timeseries(run_dir):
    #simplified with a plot config 
    jsonl_path = os.path.join(run_dir, "training_logs", "run-summary.json")
    timesteps = []
    equalities = []
    productivities = []
    actor_losses = []
    total_losses = []
    value_losses = []
    entropy = []


    with open(jsonl_path, "r") as f:
        for line in f:
            line_data = json.loads(line.strip())
            timesteps.append(line_data.get("training timestep", 0))
            equalities.append(line_data.get("equality", None))
            productivities.append(line_data.get("productivity", None))
            actor_losses.append(line_data.get("actor_loss", None))
            total_losses.append(line_data.get("total_loss", None))
            value_losses.append(line_data.get("value_loss", None))
            entropy.append(line_data.get("entropy", None))
            
            
    plot_configs = [
        ("Equality over Time",      "Equality",    equalities,    "blue",    "plot_equality_timeseries.png"),
        ("Productivity over Time",  "Productivity",productivities, "red",     "plot_productivity_timeseries.png"),
        ("Actor Loss over Time",    "Actor Loss",  actor_losses,   "green",   "plot_actor_loss_timeseries.png"),
        ("Total Loss over Time",    "Total Loss",  total_losses,   "purple",  "plot_total_loss_timeseries.png"),
        ("Value Loss over Time",    "Value Loss",  value_losses,   "orange",  "plot_value_loss_timeseries.png"),
        ("Entropy over Time",       "Entropy",     entropy,      "brown",   "plot_entropy_timeseries.png"),
    ]
            

    for title, y_label, y_values, color, filename in plot_configs:
        plt.figure(figsize=(7,5))
        plt.plot(timesteps, y_values, marker='o', label=y_label, color=color)
        plt.title(title)
        plt.xlabel("Training Timestep")
        plt.ylabel(y_label)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    

plot_run_timeseries("run_0")
plot_all_evaluations("run_0")