import os
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

log_dir = "~/blades_results/fedavg_blades"
# Expand the tilde character to your home directory
log_dir = os.path.expanduser(log_dir)

results = []
for subdir in os.listdir(log_dir):
    subdir_path = os.path.join(log_dir, subdir)
    if os.path.isdir(subdir_path):
        params_file = os.path.join(subdir_path, "params.pkl")
        results_file = os.path.join(subdir_path, "result.json")

        with open(params_file, "rb") as f:
            params = pickle.load(f)

        # breakpoint()
        with open(results_file, "r") as f:
            result_log = f.readlines()

        last_line = result_log[-1]
        final_result = json.loads(last_line)
        trial_item = {
            "Aggregator": params["server_config"]["aggregator"]["type"],
            "Momentum": params["server_config"]["optimizer"]["momentum"],
            "num_malicious_clients": params["num_malicious_clients"],
            "Accuracy": final_result["acc_top_1"],
        }
        results.append(trial_item)
results_df = pd.DataFrame(results)
# print(results_df)

grid = sns.FacetGrid(
    results_df, col="Aggregator", hue="Momentum", sharey=True, sharex=True
)
grid.map(sns.lineplot, "num_malicious_clients", "Accuracy")
grid.set(xlim=(0, 15))
grid.add_legend()
# Add a title to each facet
# for ax in grid.axes.flat:
#     ax.set_title(f"Momentum = {ax.get_xlim()[0]}")

# Show the plot
plt.show()
