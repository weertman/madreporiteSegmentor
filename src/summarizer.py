# summarizer.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import yaml
import numpy as np

def flatten_dict(d, parent_key='', sep='.'):
    """
    Recursively flattens a nested dictionary, producing a dict
    whose keys are joined by `sep`.
    Example:
        {
          'model': {'backbone': 'pvt_v2_b3', 'num_classes': 1},
          'training': {'lr': 1e-4}
        }
    becomes:
        {
          'model.backbone': 'pvt_v2_b3',
          'model.num_classes': 1,
          'training.lr': 0.0001
        }
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def summarize_experiment(experiments_base):
    """
    Summarize all results from each experiment run directory.

    1) Reads each run's `results.csv` and merges into a single `experiment_results.csv`.
       Adds flattened config.yaml columns if config.yaml is found.

    2) Creates bar charts of (loss, dice, iou, pmim) for (train, val, test).

    3) Rank-orders the test rows by dice => experiment_results_test_sorted.csv.

    4) Performs a simple correlation / group-diff analysis of config parameters
       vs. test dice => param_effect_analysis.csv.
    """
    experiments_base = Path(experiments_base)
    run_dirs = [d for d in experiments_base.iterdir() if d.is_dir()]

    # Collect all results
    all_results = []
    for run_dir in run_dirs:
        results_file = run_dir / "results.csv"
        config_file = run_dir / "config.yaml"

        if not results_file.exists():
            print(f"No results.csv found in {run_dir}, skipping.")
            continue

        # Load the results
        df = pd.read_csv(results_file)

        # Add experiment_name
        df["experiment_name"] = run_dir.name

        # Load & flatten config
        if config_file.exists():
            with open(config_file, 'r') as cf:
                run_config = yaml.safe_load(cf)
            flat_config = flatten_dict(run_config)
            for k, v in flat_config.items():
                df[k] = v
        else:
            print(f"No config.yaml found in {run_dir}, so no extra config columns added.")

        all_results.append(df)

    # If no data found, stop
    if not all_results:
        print("No experiments with results found. Nothing to summarize.")
        return

    # Concatenate
    combined = pd.concat(all_results, ignore_index=True, sort=False)

    # Save combined results
    combined_results_file = experiments_base / "experiment_results.csv"
    combined.to_csv(combined_results_file, index=False)
    print(f"Combined experiment results saved to {combined_results_file}")

    # ---- 1) Create bar plots for each split & metric ----
    # We'll look for these metrics if they exist:
    # You can include or remove metrics as needed.
    all_possible_metrics = ["loss", "dice", "iou", "pmim"]
    splits = ["train", "val", "test"]

    # Check which metrics are actually in the combined df
    existing_metrics = [m for m in all_possible_metrics if m in combined.columns]

    for split in splits:
        split_data = combined[combined["split"] == split]
        for metric in existing_metrics:
            # Group by experiment name
            group_means = split_data.groupby("experiment_name")[metric].mean().reset_index()
            # Sort by metric
            group_means = group_means.sort_values(metric, ascending=True)

            plt.figure(figsize=(10, 6))
            plt.barh(group_means["experiment_name"], group_means[metric])
            plt.xlabel(metric.capitalize())
            plt.ylabel("Experiment")
            plt.title(f"{split.capitalize()} {metric.capitalize()} by Experiment")
            plt.tight_layout()

            plot_file = experiments_base / f"{split}_{metric}_comparison.png"
            plt.savefig(plot_file)
            plt.close()
            print(f"Saved plot: {plot_file}")

    # ---- 2) Rank-order test rows by dice, if dice exists ----
    if "dice" in combined.columns:
        test_df = combined[combined["split"] == "test"].copy()
        if not test_df.empty:
            test_sorted = test_df.sort_values(by="dice", ascending=False)
            sorted_file = experiments_base / "experiment_results_test_sorted.csv"
            test_sorted.to_csv(sorted_file, index=False)
            print(f"Test results sorted by dice saved to {sorted_file}")
        else:
            print("No rows with split=='test' found, skipping test-sorted file.")
    else:
        print("No 'dice' column found, skipping rank-ordering by dice.")

    # ---- 3) Basic param-effect analysis on test dice ----
    # We'll do numeric correlation for numeric columns,
    # group mean difference for categorical.
    def analyze_param_effects(df, metric="dice"):
        """
        Return a DataFrame showing correlation or group-diff with the given metric.
        - numeric => correlation
        - categorical => max-min difference in group means
        """
        if metric not in df.columns or df.empty:
            return pd.DataFrame()

        param_effect = []
        # Columns to skip for analysis (these are the main result columns + experiment naming)
        skip_cols = {"split", "loss", "dice", "iou", "pmim", "experiment_name"}

        for col in df.columns:
            if col in skip_cols or col == metric:
                continue
            # If the column is entirely NaN or has only one unique value, skip
            if df[col].nunique(dropna=True) <= 1:
                continue

            # Distinguish numeric from categorical
            if pd.api.types.is_numeric_dtype(df[col]):
                # correlation
                corr_val = df[metric].corr(df[col])
                param_effect.append({
                    "param": col,
                    "analysis_type": "numeric_correlation",
                    "value": corr_val,
                    "abs_value": abs(corr_val)
                })
            else:
                # treat as categorical
                group_means = df.groupby(col)[metric].mean()
                if len(group_means) > 1:
                    diff_val = group_means.max() - group_means.min()
                else:
                    diff_val = np.nan
                param_effect.append({
                    "param": col,
                    "analysis_type": "categorical_group_diff",
                    "value": diff_val,
                    "abs_value": abs(diff_val) if pd.notna(diff_val) else np.nan
                })

        param_effect_df = pd.DataFrame(param_effect)
        if not param_effect_df.empty:
            # Sort by largest absolute effect
            param_effect_df.sort_values("abs_value", ascending=False, inplace=True)

        return param_effect_df

    if "dice" in combined.columns:
        # Only analyze test rows (final performance)
        test_df = combined[combined["split"] == "test"].copy()
        param_effect_df = analyze_param_effects(test_df, metric="dice")
        if not param_effect_df.empty:
            param_analysis_file = experiments_base / "param_effect_analysis.csv"
            param_effect_df.to_csv(param_analysis_file, index=False)
            print(f"Parameter effect analysis saved to {param_analysis_file}")
        else:
            print("No parameter effect analysis produced (maybe no test rows or no dice?).")
    else:
        print("No 'dice' column found in final results. Skipping param analysis.")

    print("Summary and plots completed.")

