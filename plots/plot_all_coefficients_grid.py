import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.gridspec import GridSpec


def thousands_formatter(x, pos):
    return f"{int(x/1000)}k" if x != 0 else "0"


def generate_all_pairs(pair_dict):
    results = {}
    print(type(pair_dict))
    for name, pair in pair_dict.items():
        results.update(
            {
                name: [
                    {
                        "file_path1": "data/llm_s.csv",
                        "file_path2": "data/lreg_s.csv",
                        "title": "Surprisal",
                        "coeff_cols1": [
                            f"Name: GPT2-L: {pair} - llm_coefficients/beta_s_trg",
                            f"Name: GPT2-M: {pair} - llm_coefficients/beta_s_trg",
                            f"Name: GPT2-S: {pair} - llm_coefficients/beta_s_trg",
                        ],
                        "coeff_cols2": [
                            f"Name: GPT2-L: {pair} - linear_regression_coefficients_trg/surprisal",
                            f"Name: GPT2-M: {pair} - linear_regression_coefficients_trg/surprisal",
                            f"Name: GPT2-S: {pair} - linear_regression_coefficients_trg/surprisal",
                        ],
                    },
                    {
                        "file_path1": "data/llm_wl.csv",
                        "file_path2": "data/lreg_wl.csv",
                        "title": "Word Length",
                        "coeff_cols1": [
                            f"Name: GPT2-L: {pair} - llm_coefficients/beta_l_trg",
                            f"Name: GPT2-M: {pair} - llm_coefficients/beta_l_trg",
                            f"Name: GPT2-S: {pair} - llm_coefficients/beta_l_trg",
                        ],
                        "coeff_cols2": [
                            f"Name: GPT2-L: {pair} - linear_regression_coefficients_trg/word_length",
                            f"Name: GPT2-M: {pair} - linear_regression_coefficients_trg/word_length",
                            f"Name: GPT2-S: {pair} - linear_regression_coefficients_trg/word_length",
                        ],
                    },
                    {
                        "file_path1": "data/llm_f.csv",
                        "file_path2": "data/lreg_f.csv",
                        "title": "Frequency",
                        "coeff_cols1": [
                            f"Name: GPT2-L: {pair} - llm_coefficients/beta_f_trg",
                            f"Name: GPT2-M: {pair} - llm_coefficients/beta_f_trg",
                            f"Name: GPT2-S: {pair} - llm_coefficients/beta_f_trg",
                        ],
                        "coeff_cols2": [
                            f"Name: GPT2-L: {pair} - linear_regression_coefficients_trg/word_frequency",
                            f"Name: GPT2-M: {pair} - linear_regression_coefficients_trg/word_frequency",
                            f"Name: GPT2-S: {pair} - linear_regression_coefficients_trg/word_frequency",
                        ],
                    },
                    {
                        "file_path1": "data/llm_const.csv",
                        "file_path2": "data/lreg_const.csv",
                        "title": "Bias",
                        "coeff_cols1": [
                            f"Name: GPT2-L: {pair} - llm_coefficients/beta_0_trg",
                            f"Name: GPT2-M: {pair} - llm_coefficients/beta_0_trg",
                            f"Name: GPT2-S: {pair} - llm_coefficients/beta_0_trg",
                        ],
                        "coeff_cols2": [
                            f"Name: GPT2-L: {pair} - linear_regression_coefficients_trg/const",
                            f"Name: GPT2-M: {pair} - linear_regression_coefficients_trg/const",
                            f"Name: GPT2-S: {pair} - linear_regression_coefficients_trg/const",
                        ],
                    },
                ]
            }
        )
    return results


plt.rcParams["mathtext.fontset"] = "cm"  # Computer Modern, matches LaTeX default
plt.rcParams["font.family"] = "sans-serif"

plt.rcParams.update(
    {
        "figure.constrained_layout.use": False,
        "font.size": 20,
        "axes.edgecolor": "lightgray",
        "xtick.color": "gray",
        "ytick.color": "gray",
        "axes.labelcolor": "gray",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.major.size": 4,
        "ytick.minor.size": 2,
        "xtick.major.pad": 1,
        "ytick.major.pad": 1,
        "grid.alpha": 0.3,
    }
)

colors1 = ["#BBDEFB", "#C8E6C9", "#E1BEE7"]
colors2 = ["#0D47A1", "#1B5E20", "#4A148C"]

# colors2 = ["#178460", "#AC7822", "#5F38A3"]
# colors1 = ["#A12864", "#1f77b4", "cyan", "magenta"]  # Colors for GPT2-L
# colors2 = ["#2ca02c", "#5BC5DB", "purple", "brown"]  # Colors for Regresso


def process_and_plot_pair(
    ax,
    ax2,
    file_path1,
    file_path2,
    title,
    coeff_cols1,
    coeff_cols2,
    colors1,
    colors2,
    showx=True,
    showtite=True,
):
    data1 = pd.read_csv(file_path1)
    data2 = pd.read_csv(file_path2)
    seeds = ["42", "8", "64"]
    ticklabelsize = 13

    for idx, coeff_col in enumerate(coeff_cols1):
        all_relevant_data = []

        for seed in seeds:
            new_coeff_col = coeff_col.replace(" -", f"_{seed} -").lstrip("Name: ")
            if new_coeff_col in data1.columns:
                relevant_data = data1[["Step", new_coeff_col]].dropna()
                relevant_data = relevant_data.rename(columns={new_coeff_col: "Value"})
                all_relevant_data.append(relevant_data)
        if all_relevant_data:
            combined_data = pd.concat(all_relevant_data)
            print("Combined Data")
            combined_data["Step"] = combined_data["Step"] * 50
            print(combined_data)
            grouped_data = combined_data.groupby("Step")["Value"].mean().reset_index()
            data = grouped_data["Value"]
            smoothed_data = np.zeros_like(data)
            smoothed_data[0] = data.iloc[0]
            smoothed_data[-1] = data.iloc[-1]
            for i in range(1, len(data) - 1):
                smoothed_data[i] = np.mean(data.iloc[i - 1 : i + 4])

            ax.plot(
                grouped_data["Step"],
                data,
                "-o",
                label="",
                color=colors1[idx % len(colors1)],
                linewidth=0.8,
                markersize=3,
                alpha=0.4,
            )

            ax.plot(
                grouped_data["Step"],
                smoothed_data,
                "-o",
                label=r"$\mathrm{\beta_\theta}$"
                + f" {new_coeff_col.split(' - ')[0].split(':')[0]}",
                color=colors1[idx % len(colors1)],
                linewidth=0.8,
                markersize=3,
                alpha=1,
            )
    ax.set_xlabel("", fontsize=20)
    ax.set_ylabel("", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=ticklabelsize)
    ax.set_xlim(0, 5000)

    formatter = FuncFormatter(thousands_formatter)
    ax.xaxis.set_major_formatter(formatter)
    # Reduce number of ticks
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    if showx == False:
        ax.set(xticklabels=[])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

    ax.grid(True)

    for idx, coeff_col in enumerate(coeff_cols2):
        all_relevant_data2 = []
        for seed in seeds:
            new_coeff_col = coeff_col.replace(" -", f"_{seed} -").lstrip("Name: ")
            if new_coeff_col in data2.columns:
                relevant_data = data2[["Step", new_coeff_col]].dropna()
                relevant_data = relevant_data.rename(columns={new_coeff_col: "Value"})
                all_relevant_data2.append(relevant_data)
        if all_relevant_data2:
            combined_data2 = pd.concat(all_relevant_data2)
            combined_data2["Step"] = combined_data2["Step"] * 50
            grouped_data2 = combined_data2.groupby("Step")["Value"].mean().reset_index()

            # min_step2 = min(min_step2, grouped_data2["Step"].min())
            # max_step2 = max(max_step2, grouped_data2["Step"].max())
            # min_value2 = min(min_value2, grouped_data2["Value"].min())
            # max_value2 = max(max_value2, grouped_data2["Value"].max())
            val_data2 = grouped_data2["Value"]

            smoothed_data2 = np.zeros_like(val_data2)
            smoothed_data2[0] = val_data2.iloc[0]
            smoothed_data2[-1] = val_data2.iloc[-1]
            for i in range(1, len(val_data2) - 1):
                smoothed_data2[i] = np.mean(val_data2.iloc[i - 1 : i + 4])

            ax2.plot(
                grouped_data2["Step"],
                val_data2,
                "-s",
                label="",
                color=colors2[idx % len(colors2)],
                linewidth=1,
                markersize=2,
                alpha=0.2,
            )

            # Plot the smoothed data on top
            ax2.plot(
                grouped_data2["Step"],
                smoothed_data2,
                "-s",
                label=r"$\mathrm{\beta_R}$"
                + f" {new_coeff_col.split(' - ')[0].split(':')[0]}",
                color=colors2[idx % len(colors2)],
                linewidth=1,
                markersize=2,
                alpha=1,
            )

    if showtite:
        # ax2.set_title(title, fontsize=28)
        pos1 = ax.get_position()
        # midpoint_x = (pos1.x0 + pos2.x0) / 2
        x_shift = (
            1.7
            if title == "Word Length"
            else 1.3 if title == "Bias" else 1.6 if title == "Frequency" else 1.5
        )
        ax.set_title(title, fontsize=22, y=1.04, pad=5, x=x_shift, loc="right")
        """
        ax2.text(
            midpoint_x,
            0.88,
            title,
            ha="center",
            va="center",
            fontsize=28,
            transform=fig.transFigure,
        )
        """
    ax2.set_xlabel("", fontsize=18)
    ax2.set_ylabel("", fontsize=18)
    ax2.tick_params(axis="both", which="major", labelsize=ticklabelsize)
    ax2.set_xlim(0, 5000)

    formatter = FuncFormatter(thousands_formatter)
    ax2.xaxis.set_major_formatter(formatter)
    # Reduce number of ticks
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=3))
    if showx == False:
        ax.set(xticklabels=[])
        ax2.set(xticklabels=[])

    ax2.yaxis.set_major_locator(MaxNLocator(nbins=3))

    ax2.grid(True)


label_dict = {
    "dd": "D ➛ D",
    "pd": "P ➛ D",
    "zd": "Z ➛ D",
    "dp": "D ➛ P",
    "pp": "P ➛ P",
    "pz": "P ➛ Z",
    "dz": "D ➛ Z",
    "zp": "Z ➛ P",
    "zz": "Z ➛ Z",
}

all_pairs = generate_all_pairs(label_dict)
print(all_pairs)

fig = plt.figure(figsize=(20, 18))

gs = GridSpec(
    9,
    8,
    figure=fig,
    width_ratios=[1] * 8,
    height_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1],
    wspace=0.35,
    hspace=0.1,
)

axs = []
for row in range(9):
    for col in range(8):
        axs.append(fig.add_subplot(gs[row, col]))

index = 0
for row_count, (label_name, pairs) in enumerate(all_pairs.items()):

    for i, pair in enumerate(pairs):
        process_and_plot_pair(
            axs[index],
            axs[index + 1],
            pair["file_path1"],
            pair["file_path2"],
            pair["title"],
            pair["coeff_cols1"],
            pair["coeff_cols2"],
            colors1,
            colors2,
            showx=True if label_name == "zz" else False,
            showtite=True if label_name == "dd" else False,
        )
        index += 2

    handles, labels = [], []
    for ax in axs:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    # y_pos = 1 - ((row_count) + 0.3) / 9
    ax_position = axs[row_count * 8].get_position()
    y_pos = (ax_position.y0 + ax_position.y1) / 2
    # if label_name == "pp":
    #    y_pos += 0.1
    # elif label_name == "dd":
    #    y_pos -= 0.1
    fig.text(
        0.10,
        y_pos,
        f"{label_dict[label_name]}",
        ha="center",
        va="center",
        fontsize=16,
        rotation="vertical",
    )


leg = fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=6,
    fontsize=20,
    frameon=False,
    bbox_to_anchor=(0.5, 0.93),
)
for lh in leg.legendHandles:
    lh.set_alpha(1)
    lh.set_markersize(10)
for line in leg.get_lines():
    line.set_linewidth(3.0)

fig.text(0.5, 0.085, "Steps", ha="center", fontsize=20)
plt.savefig(f"all_coefficients_grid.pdf", dpi=300, bbox_inches="tight")
