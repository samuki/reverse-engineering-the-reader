import pandas as pd

dll_stepwise_out = pd.read_csv("dll_stepwise_out.csv")
dll_start_values = dll_stepwise_out[dll_stepwise_out["Step"] == 0].copy()
dll_start_values = (
    dll_start_values.groupby("Experiment")
    .agg({"Mean": "mean", "SEM": "mean"})
    .reset_index()
)

dll_max_indices = dll_stepwise_out.groupby("Experiment")["Mean"].idxmax()
dll_max_values = dll_stepwise_out.loc[
    dll_max_indices, ["Experiment", "Mean", "SEM"]
].set_index("Experiment")

dll_values = pd.merge(
    dll_start_values, dll_max_values, on="Experiment", suffixes=("_start", "_max")
)

dll_values["Model"] = dll_values["Experiment"].str.extract(r"(GPT2-[LMS])")[0]
dll_values["Source"] = dll_values["Experiment"].str.extract(r": ([DPZ])")[0]
dll_values["Target"] = dll_values["Experiment"].str.extract(r"➛ ([DPZ])")[0]

dll_values = dll_values.sort_values(by=["Model", "Target"])

print(dll_values)

dll_values["Mean_max"] = dll_values["Mean_max"] * 100
dll_values["SEM_max"] = dll_values["SEM_max"] * 100
dll_values["Mean_start"] = dll_values["Mean_start"] * 100

dll_values["% Increase"] = (
    (dll_values["Mean_max"] - dll_values["Mean_start"]) / dll_values["Mean_start"]
) * 100

dll_values["Experiment"] = dll_values["Experiment"].str.replace("➛", "$\\rightarrow$")

"""
dll_values["Mean_start"] = dll_values["Mean_start"].apply(lambda x: f"{x:.3f}")

dll_values["Mean_max"] = (
    dll_values["Mean_max"].apply(lambda x: f"{x:.3f}")
    + "_{"
    + dll_values["SEM_max"].apply(lambda x: f"{x:.3f}")
    + "}"
)
"""

# dll_values["% Increase"] = dll_values["% Increase"].apply(lambda x: f"{x:.3f}")

dll_values = dll_values.drop(columns=["Model", "Source", "Target"])

dll_values["Model"] = dll_values["Experiment"].apply(lambda x: x.split(":")[0])
dll_values["Data"] = dll_values["Experiment"].apply(lambda x: x.split(":")[1].strip())

latex_table = """
\\begin{table}[h!] 
    \\small
    \\centering
    \\begin{tabular}{l|ccc|ccc|ccc} 
    \\toprule 
    \\multicolumn{1}{c}{} & \\multicolumn{3}{c}{GPT2-L} & \\multicolumn{3}{c}{GPT2-M} & \\multicolumn{3}{c}{GPT2-S} \\\\
    Data &  $\\dll^{\\text{start}}$ &  $\\dll^{\\text{max}}$ & \\% Increase 
              &  $\\dll^{\\text{start}}$ &  $\\dll^{\\text{max}}$ & \\% Increase 
              &  $\\dll^{\\text{start}}$ &  $\\dll^{\\text{max}}$ & \\% Increase \\\\ 
     \\midrule  
"""

data_patterns = dll_values["Data"].unique()
for data_pattern in data_patterns:
    row_L = dll_values[
        (dll_values["Model"] == "GPT2-L") & (dll_values["Data"] == data_pattern)
    ].iloc[0]
    row_M = dll_values[
        (dll_values["Model"] == "GPT2-M") & (dll_values["Data"] == data_pattern)
    ].iloc[0]
    row_S = dll_values[
        (dll_values["Model"] == "GPT2-S") & (dll_values["Data"] == data_pattern)
    ].iloc[0]
    latex_table += (
        f"    {data_pattern} & "
        f"${row_L['Mean_start']:.2f}$ & ${row_L['Mean_max']:.2f} \\pm {row_L['SEM_max']:.2f}$ & {row_L['% Increase']:.2f} & "
        f"${row_M['Mean_start']:.2f}$ & ${row_M['Mean_max']:.2f} \\pm {row_M['SEM_max']:.2f}$ & {row_M['% Increase']:.2f} & "
        f"${row_S['Mean_start']:.2f}$ & ${row_S['Mean_max']:.2f} \\pm {row_S['SEM_max']:.2f}$ & {row_S['% Increase']:.2f} \\\\ \n"
    )
latex_table += """
    \\bottomrule
    \\end{tabular}
    \\caption{Mean start and maximum $\\dll$ values including standard errors across all experiments rounded to two digits.}
    \\label{tab:dll-table}
\\end{table}
"""
print(latex_table)
