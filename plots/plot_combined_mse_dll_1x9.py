import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


def thousands_formatter(x, pos):
    return f"{int(x/1000)}k" if x != 0 else "0"


def formatter_y(x, pos):
    return f"{round(float(x*100),1)}" if x != 0 else "0"


# Update params here
plt.rcParams["mathtext.fontset"] = "cm"  # Computer Modern, matches LaTeX default
plt.rcParams["font.family"] = "sans-serif"

plt.rcParams.update(
    {
        "figure.constrained_layout.use": True,
        "font.size": 12,
        "axes.edgecolor": "#696969",
        "xtick.color": "#696969",
        "ytick.color": "#696969",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 6,
        "xtick.minor.size": 4,
        "ytick.major.size": 6,
        "ytick.minor.size": 4,
        "xtick.major.pad": 5,
        "xtick.minor.pad": 5,
        "ytick.major.pad": 1,
        "ytick.minor.pad": 1,
    }
)

mse_data = pd.read_csv("mse_stepwise_out.csv")
mse_pivoted_data = mse_data.pivot(index="Step", columns="Experiment", values="Mean")
mse_sem_data = mse_data.pivot(index="Step", columns="Experiment", values="SEM")
mse_pivoted_data = mse_pivoted_data.replace([np.inf, -np.inf], np.nan)
mse_sem_data = mse_sem_data.replace([np.inf, -np.inf], np.nan)


experiments_Z_Z = [col for col in mse_pivoted_data.columns if "Z ➛ Z" in col]
experiments_D_Z = [col for col in mse_pivoted_data.columns if "D ➛ Z" in col]
experiments_Z_D = [col for col in mse_pivoted_data.columns if "Z ➛ D" in col]
experiments_P_Z = [col for col in mse_pivoted_data.columns if "P ➛ Z" in col]
experiments_Z_P = [col for col in mse_pivoted_data.columns if "Z ➛ P" in col]

experiments_D_D = [col for col in mse_pivoted_data.columns if "D ➛ D" in col]
experiments_D_P = [col for col in mse_pivoted_data.columns if "D ➛ P" in col]
experiments_P_D = [col for col in mse_pivoted_data.columns if "P ➛ D" in col]

experiments_P_P = [col for col in mse_pivoted_data.columns if "P ➛ P" in col]


dll_data = pd.read_csv("dll_stepwise_out.csv")
dll_pivoted_data = dll_data.pivot(index="Step", columns="Experiment", values="Mean")
dll_sem_data = dll_data.pivot(index="Step", columns="Experiment", values="SEM")
dll_pivoted_data = dll_pivoted_data.replace([np.inf, -np.inf], np.nan)
dll_sem_data = dll_sem_data.replace([np.inf, -np.inf], np.nan)

dll_experiments_Z_Z = [col for col in dll_pivoted_data.columns if "Z ➛ Z" in col]
dll_experiments_D_Z = [col for col in dll_pivoted_data.columns if "D ➛ Z" in col]
dll_experiments_Z_D = [col for col in dll_pivoted_data.columns if "Z ➛ D" in col]
dll_experiments_P_Z = [col for col in dll_pivoted_data.columns if "P ➛ Z" in col]
dll_experiments_Z_P = [col for col in dll_pivoted_data.columns if "Z ➛ P" in col]
dll_experiments_D_D = [col for col in dll_pivoted_data.columns if "D ➛ D" in col]
dll_experiments_D_P = [col for col in dll_pivoted_data.columns if "D ➛ P" in col]
dll_experiments_P_D = [col for col in dll_pivoted_data.columns if "P ➛ D" in col]
dll_experiments_P_P = [col for col in dll_pivoted_data.columns if "P ➛ P" in col]


# Experiment-Color dicts
color_dict_Z_Z = {
    "GPT2-L: Z ➛ Z": "#1f77b4",
    "GPT2-M: Z ➛ Z": "#2ca02c",
    "GPT2-S: Z ➛ Z": "#9467bd",
}

color_dict_Z_D = {
    "GPT2-L: Z ➛ D": "#1f77b4",
    "GPT2-M: Z ➛ D": "#2ca02c",
    "GPT2-S: Z ➛ D": "#9467bd",
}

color_dict_Z_P = {
    "GPT2-L: Z ➛ P": "#1f77b4",
    "GPT2-M: Z ➛ P": "#2ca02c",
    "GPT2-S: Z ➛ P": "#9467bd",
}

color_dict_D_Z = {
    "GPT2-L: D ➛ Z": "#1f77b4",
    "GPT2-M: D ➛ Z": "#2ca02c",
    "GPT2-S: D ➛ Z": "#9467bd",
}

color_dict_D_D = {
    "GPT2-L: D ➛ D": "#1f77b4",
    "GPT2-M: D ➛ D": "#2ca02c",
    "GPT2-S: D ➛ D": "#9467bd",
}

color_dict_D_P = {
    "GPT2-L: D ➛ P": "#1f77b4",
    "GPT2-M: D ➛ P": "#2ca02c",
    "GPT2-S: D ➛ P": "#9467bd",
}

color_dict_P_Z = {
    "GPT2-L: P ➛ Z": "#1f77b4",
    "GPT2-M: P ➛ Z": "#2ca02c",
    "GPT2-S: P ➛ Z": "#9467bd",
}

color_dict_P_D = {
    "GPT2-L: P ➛ D": "#1f77b4",
    "GPT2-M: P ➛ D": "#2ca02c",
    "GPT2-S: P ➛ D": "#9467bd",
}

color_dict_P_P = {
    "GPT2-L: P ➛ P": "#1f77b4",
    "GPT2-M: P ➛ P": "#2ca02c",
    "GPT2-S: P ➛ P": "#9467bd",
}


# Plot experiments
def plot_experiments(ax, experiments, val, color_dict, mse=True):
    for experiment in experiments:
        if mse:
            pivoted_data = mse_pivoted_data
            sem_data = mse_sem_data
        else:
            pivoted_data = dll_pivoted_data
            sem_data = dll_sem_data
        mean_values = pivoted_data[experiment]
        sem_values = sem_data[experiment]

        # Remove NaN
        mask = ~mean_values.isna()
        steps = pivoted_data.index[mask] * 50
        mean_values = mean_values[mask]
        sem_values = sem_values[mask]
        ax.plot(
            steps[:1],
            mean_values[:1],
            "*",
            label="_nolegend_",
            color=color_dict.get(experiment, "black"),
            markersize=8,
            alpha=1,
            zorder=1000,
        )
        ax.plot(
            steps,
            mean_values,
            "-o",
            label=experiment,
            color=color_dict.get(experiment, "black"),
            linewidth=1,
            markersize=2,
            alpha=0.8,
        )
        ax.fill_between(
            steps,
            mean_values - sem_values,
            mean_values + sem_values,
            color=color_dict.get(experiment, "black"),
            alpha=0.05,
        )

    formatter = FuncFormatter(thousands_formatter)
    formattery = FuncFormatter(formatter_y)
    ax.xaxis.set_major_formatter(formatter)
    if not mse:
        ax.yaxis.set_major_formatter(formattery)
    if mse:
        ax.set_xticklabels([])
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_xlim(-150, 5000)
    ax.grid(True, linestyle="--", alpha=0.7)
    # ax.legend(fontsize=10, loc="best")


fig = plt.figure(figsize=(20, 5))
gs = GridSpec(
    2,
    9,
    figure=fig,
    width_ratios=[1] * 9,
    height_ratios=[1, 1],
    wspace=0.000001,
    hspace=0.001,
)

axs = []
for row in range(2):
    for col in range(9):
        axs.append(fig.add_subplot(gs[row, col]))

# Plot each experiment group
plot_experiments(axs[0], experiments_D_D, "D_D", color_dict=color_dict_D_D)
plot_experiments(axs[1], experiments_Z_D, "Z_D", color_dict=color_dict_Z_D)
plot_experiments(axs[2], experiments_P_D, "P_D", color_dict=color_dict_P_D)
plot_experiments(axs[3], experiments_D_Z, "D_Z", color_dict=color_dict_D_Z)
plot_experiments(axs[4], experiments_Z_Z, "Z_Z", color_dict=color_dict_Z_Z)
plot_experiments(axs[5], experiments_P_Z, "P_Z", color_dict=color_dict_P_Z)
plot_experiments(axs[6], experiments_D_P, "D_P", color_dict=color_dict_D_P)
plot_experiments(axs[7], experiments_Z_P, "Z_P", color_dict=color_dict_Z_P)
plot_experiments(axs[8], experiments_P_P, "P_P", color_dict=color_dict_P_P)


plot_experiments(
    axs[9], dll_experiments_D_D, "D_D", color_dict=color_dict_D_D, mse=False
)
plot_experiments(
    axs[10], dll_experiments_Z_D, "Z_D", color_dict=color_dict_Z_D, mse=False
)
plot_experiments(
    axs[11], dll_experiments_P_D, "P_D", color_dict=color_dict_P_D, mse=False
)
plot_experiments(
    axs[12], dll_experiments_D_Z, "D_Z", color_dict=color_dict_D_Z, mse=False
)
plot_experiments(
    axs[13], dll_experiments_Z_Z, "Z_Z", color_dict=color_dict_Z_Z, mse=False
)
plot_experiments(
    axs[14], dll_experiments_P_Z, "P_Z", color_dict=color_dict_P_Z, mse=False
)
plot_experiments(
    axs[15], dll_experiments_D_P, "D_P", color_dict=color_dict_D_P, mse=False
)
plot_experiments(
    axs[16], dll_experiments_Z_P, "Z_P", color_dict=color_dict_Z_P, mse=False
)
plot_experiments(
    axs[17], dll_experiments_P_P, "P_P", color_dict=color_dict_P_P, mse=False
)

# Add titles for each plot
titles = [
    "D ➛ D",
    "Z ➛ D",
    "P ➛ D",
    "D ➛ Z",
    "Z ➛ Z",
    "P ➛ Z",
    "D ➛ P",
    "Z ➛ P",
    "P ➛ P",
]
for ax, title in zip(axs, titles):
    ax.set_title(title, fontsize=18)

gpt2_l = Line2D([0], [0], label="GPT2-L", color="#1f77b4", linewidth=4)
gpt2_m = Line2D([0], [0], label="GPT2-M", color="#2ca02c", linewidth=4)
gpt2_s = Line2D([0], [0], label="GPT2-S", color="#9467bd", linewidth=4)

fig.legend(
    handles=[gpt2_l, gpt2_m, gpt2_s],
    loc="upper center",
    ncol=3,
    fontsize=20,
    bbox_to_anchor=(0.5, 1.12),
    frameon=False,
)

fig.text(0.5, -0.03, "Steps", ha="center", fontsize=18)
fig.text(
    -0.01,
    0.28,
    r"$\mathrm{\Delta_{llh} (10^{-2} \, nats)}$",
    va="center",
    rotation="vertical",
    fontsize=20,
)
fig.text(
    -0.01,
    0.7,
    "MSE",
    va="center",
    rotation="vertical",
    fontsize=18,
)

# plt.tight_layout(rect=[0.02, 0.02, 1, 0.95])
# plt.subplots_adjust(wspace=0.3, hspace=0.2)

plt.savefig("mean_mse_dll_1x9.pdf", dpi=300, bbox_inches="tight")
