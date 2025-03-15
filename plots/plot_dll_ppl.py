import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
from matplotlib.ticker import FuncFormatter
import os


def formatter_y(x, pos):
    return f"{round(float(x*100),1)}" if x != 0 else "0"


plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "sans-serif"

plt.rcParams.update(
    {
        "font.size": 16,
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
        "xtick.major.pad": 15,
        "xtick.minor.pad": 15,
        "ytick.major.pad": 15,
        "ytick.minor.pad": 15,
    }
)

ppl_stepwise_out = pd.read_csv("ppl_stepwise_out.csv")
dll_stepwise_out = pd.read_csv("dll_stepwise_out.csv")
dll_start_values = (
    dll_stepwise_out[dll_stepwise_out["Step"] == 0].groupby("Experiment")["Mean"].mean()
)
dll_max_indices = dll_stepwise_out.groupby("Experiment")["Mean"].idxmax()
dll_max_values = dll_stepwise_out.loc[
    dll_max_indices, ["Experiment", "Mean"]
].set_index("Experiment")["Mean"]

print("Max indices ", dll_max_indices)

dll_max_steps = dll_stepwise_out.loc[
    dll_max_indices, ["Experiment", "Step", "Mean"]
].set_index("Experiment")["Step"]

print("Max Steps", dll_max_steps)

perplexity_start_values = (
    ppl_stepwise_out[ppl_stepwise_out["Step"] == 0].groupby("Experiment")["Mean"].mean()
)

perplexity_max_values = ppl_stepwise_out.loc[
    dll_max_indices, ["Experiment", "Mean", "Step"]
].set_index("Experiment")["Mean"]

print("ppl max values ", perplexity_max_values)

ppl_max_values = ppl_stepwise_out.set_index(["Experiment", "Step"]).loc[
    pd.MultiIndex.from_tuples(zip(dll_max_steps.index, dll_max_steps))
]["Mean"]

ppl_max_values = ppl_max_values.reset_index().set_index("Experiment")["Mean"]
perplexity_max_values = ppl_max_values.rename("ppl max values")

print("Final Perplexity Max Values:", perplexity_max_values)

for experiment in dll_max_steps.index:
    dll_max_step = dll_max_steps[experiment]

    expected_ppl = ppl_stepwise_out[
        (ppl_stepwise_out["Experiment"] == experiment)
        & (ppl_stepwise_out["Step"] == dll_max_step)
    ]["Mean"].values[0]

    computed_ppl = ppl_max_values.loc[experiment]
    assert (
        computed_ppl == expected_ppl
    ), f"Mismatch in PPL vas for {experiment}: expected {expected_ppl}, got {computed_ppl}"

print(" assertions passed")

subset_experiments_D = [
    "GPT2-L: D ➛ D",
    "GPT2-M: D ➛ D",
    "GPT2-S: D ➛ D",
    "GPT2-L: Z ➛ D",
    "GPT2-M: Z ➛ D",
    "GPT2-S: Z ➛ D",
    "GPT2-L: P ➛ D",
    "GPT2-M: P ➛ D",
    "GPT2-S: P ➛ D",
]

subset_experiments_P = [
    "GPT2-L: D ➛ P",
    "GPT2-M: D ➛ P",
    "GPT2-S: D ➛ P",
    "GPT2-L: Z ➛ P",
    "GPT2-M: Z ➛ P",
    "GPT2-S: Z ➛ P",
    "GPT2-L: P ➛ P",
    "GPT2-M: P ➛ P",
    "GPT2-S: P ➛ P",
]
subset_experiments_Z = [
    "GPT2-L: D ➛ Z",
    "GPT2-M: D ➛ Z",
    "GPT2-S: D ➛ Z",
    "GPT2-L: Z ➛ Z",
    "GPT2-M: Z ➛ Z",
    "GPT2-S: Z ➛ Z",
    "GPT2-L: P ➛ Z",
    "GPT2-M: P ➛ Z",
    "GPT2-S: P ➛ Z",
]

combine_experiments = [
    "GPT2-L: D ➛ D",
    "GPT2-M: D ➛ D",
    "GPT2-S: D ➛ D",
    "GPT2-L: Z ➛ Z",
    "GPT2-M: Z ➛ Z",
    "GPT2-S: Z ➛ Z",
    "GPT2-L: P ➛ P",
    "GPT2-M: P ➛ P",
    "GPT2-S: P ➛ P",
]

custom_colors_P = {
    "GPT2-L: D ➛ P": "#B0E0E6",
    "GPT2-M: D ➛ P": "#32CD32",
    "GPT2-S: D ➛ P": "#DDA0DD",
    "GPT2-L: Z ➛ P": "#4682B4",
    "GPT2-M: Z ➛ P": "#228B22",
    "GPT2-S: Z ➛ P": "#9932CC",
    "GPT2-L: P ➛ P": "#000080",
    "GPT2-M: P ➛ P": "#006400",
    "GPT2-S: P ➛ P": "#4B0082",
}

custom_colors_D = {
    "GPT2-L: D ➛ D": "#B0E0E6",
    "GPT2-M: D ➛ D": "#32CD32",
    "GPT2-S: D ➛ D": "#DDA0DD",
    "GPT2-L: Z ➛ D": "#4682B4",
    "GPT2-M: Z ➛ D": "#228B22",
    "GPT2-S: Z ➛ D": "#9932CC",
    "GPT2-L: P ➛ D": "#000080",
    "GPT2-M: P ➛ D": "#006400",
    "GPT2-S: P ➛ D": "#4B0082",
}

custom_colors_Z = {
    "GPT2-L: D ➛ Z": "#B0E0E6",
    "GPT2-M: D ➛ Z": "#32CD32",
    "GPT2-S: D ➛ Z": "#DDA0DD",
    "GPT2-L: Z ➛ Z": "#4682B4",
    "GPT2-M: Z ➛ Z": "#228B22",
    "GPT2-S: Z ➛ Z": "#9932CC",
    "GPT2-L: P ➛ Z": "#000080",
    "GPT2-M: P ➛ Z": "#006400",
    "GPT2-S: P ➛ Z": "#4B0082",
}


def plot_dll_ppl(subset_experiments, custom_colors, experiment_name):
    filtered_dll_start_values = dll_start_values[
        dll_start_values.index.isin(subset_experiments)
    ]
    filtered_dll_max_values = dll_max_values[
        dll_max_values.index.isin(subset_experiments)
    ]
    filtered_perplexity_start_values = perplexity_start_values[
        perplexity_start_values.index.isin(subset_experiments)
    ]
    filtered_perplexity_max_values = perplexity_max_values[
        perplexity_max_values.index.isin(subset_experiments)
    ]

    filtered_data = pd.DataFrame(
        {
            "DLL Start": filtered_dll_start_values,
            "DLL Max": filtered_dll_max_values,
            "Perplexity Start": filtered_perplexity_start_values,
            "Perplexity Max": filtered_perplexity_max_values,
        }
    ).reset_index()

    filtered_data["Log Perplexity Start"] = np.log(filtered_data["Perplexity Start"])
    filtered_data["Log Perplexity Max"] = np.log(filtered_data["Perplexity Max"])

    fig, ax = plt.subplots(figsize=(6.4, 3.5))

    for i, row in filtered_data.iterrows():
        color = custom_colors[row["Experiment"]]
        plt.scatter(row["Log Perplexity Start"], row["DLL Start"], color=color, s=80)
        plt.scatter(
            row["Log Perplexity Max"], row["DLL Max"], color=color, marker="x", s=80
        )
        plt.plot(
            [row["Log Perplexity Start"], row["Log Perplexity Max"]],
            [row["DLL Start"], row["DLL Max"]],
            color=color,
            linestyle="--",
            label=row["Experiment"],
            linewidth=3,
            alpha=0.8,
            zorder=-1,
        )

    plt.xlabel(r"Log Perplexity", fontsize=18)
    leg1 = plt.legend(
        loc="upper center",
        frameon=False,
        ncol=3,
        bbox_to_anchor=(0.45, 1.4),
        fontsize=13,
    )

    start_marker = mlines.Line2D(
        [], [], color="black", marker="o", linestyle="None", markersize=9, label="Start"
    )
    end_marker = mlines.Line2D(
        [],
        [],
        color="black",
        marker="x",
        linestyle="None",
        markersize=9,
        label="Max " + r"$\mathrm{\Delta_{llh}}$",
    )

    plt.gca().add_artist(
        plt.legend(
            handles=[start_marker, end_marker],
            loc="best",
            ncol=1,
            fontsize=13,
        )
    )

    ax.add_artist(leg1)
    ax.set_axisbelow(True)

    formattery = FuncFormatter(formatter_y)
    ax.yaxis.set_major_formatter(formattery)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(4))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(4))
    fig.text(
        -0.02,
        0.6,
        r"$\mathrm{\Delta_{llh} (10^{-2} \, nats)}$",
        va="center",
        rotation="vertical",
        fontsize=20,
    )
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    os.makedirs("x_y_chart", exist_ok=True)
    os.makedirs("x_y_chart/dll_ppl", exist_ok=True)
    plt.savefig(
        f"x_y_chart/dll_ppl/{experiment_name}_dll_ppl.pdf",
        dpi=300,
        bbox_inches="tight",
    )


combined_custom_colors = {**custom_colors_D, **custom_colors_P, **custom_colors_Z}

plot_dll_ppl(subset_experiments_D, custom_colors_D, "D")
plot_dll_ppl(subset_experiments_P, custom_colors_P, "P")
plot_dll_ppl(subset_experiments_Z, custom_colors_Z, "Z")

plot_dll_ppl(combine_experiments, combined_custom_colors, "Combined_in_domain")
