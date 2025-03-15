import pandas as pd
import numpy as np

file_path = "data/dll.csv"
df = pd.read_csv(file_path)

delta_llh_columns = [
    col for col in df.columns if col.endswith("linear_regression/delta_llh")
]
experiment_names = set([col.split("_")[0] for col in delta_llh_columns])

results = []
for experiment_name in experiment_names:
    relevant_cols = [col for col in delta_llh_columns if experiment_name in col]
    print(f"Experiment: {experiment_name}")
    print(f"Relevant Columns: {relevant_cols}")
    if len(relevant_cols) != 3:
        print(f"Skipping {experiment_name} due to missing columns")
        continue
    for step in df["Step"].unique():
        if step <= 100:
            step_means = []
            for col in relevant_cols:
                step_data = df[df["Step"] == step][col]
                print(f"Step Data for {col}:")
                print(step_data.head())
                if not step_data.empty:
                    step_means.append(step_data.mean())
                print(f"Step Means: {step_means}")
            assert len(step_means) == 3
            if step_means:
                step_mean = np.mean(step_means)
                step_sem = np.std(step_means, ddof=1) / np.sqrt(len(step_means))

                results.append([experiment_name, step, step_mean, step_sem])

results_df = pd.DataFrame(
    results,
    columns=[
        "Experiment",
        "Step",
        "Mean",
        "SEM",
    ],
)


output_file_path = "dll_stepwise_out.csv"
results_df.to_csv(output_file_path, index=False)
