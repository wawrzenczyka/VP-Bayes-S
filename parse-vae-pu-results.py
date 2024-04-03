# %%
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

pd.set_option("display.max_rows", 500)

root = "result"
# root = "result-cc"
results = []

for dataset in os.listdir(root):
    if not os.path.isdir(os.path.join(root, dataset)):
        continue

    for c in os.listdir(os.path.join(root, dataset)):
        if c.startswith("Exp"):
            continue

        for exp in os.listdir(os.path.join(root, dataset, c)):
            exp_num = int(exp[3:])

            try:
                if os.path.exists(
                    os.path.join(root, dataset, c, exp, "metric_values.json")
                ):
                    with open(
                        os.path.join(root, dataset, c, exp, "metric_values.json"), "r"
                    ) as f:
                        metrics = json.load(f)

                        metrics["Dataset"] = dataset
                        metrics["Experiment"] = exp_num
                        # metrics['c'] = 0.5 if 20 > (exp_num % 20) >= 10 else 0.02
                        metrics["c"] = float(c)

                        results.append(metrics)

                if os.path.exists(
                    os.path.join(root, dataset, c, exp, "metric_values_orig.json")
                ):
                    with open(
                        os.path.join(root, dataset, c, exp, "metric_values_orig.json"),
                        "r",
                    ) as f:
                        metrics = json.load(f)

                        metrics["Dataset"] = dataset
                        metrics["Experiment"] = exp_num
                        # metrics['c'] = 0.5 if 20 > (exp_num % 20) >= 10 else 0.02
                        metrics["c"] = float(c)

                        results.append(metrics)

                if os.path.exists(os.path.join(root, dataset, c, exp, "external")):
                    for method in os.listdir(
                        os.path.join(root, dataset, c, exp, "external")
                    ):
                        with open(
                            os.path.join(
                                root,
                                dataset,
                                c,
                                exp,
                                "external",
                                method,
                                "metric_values.json",
                            ),
                            "r",
                        ) as f:
                            metrics = json.load(f)

                            metrics["Dataset"] = dataset
                            metrics["Experiment"] = exp_num
                            # metrics['c'] = 0.5 if 20 > (exp_num % 20) >= 10 else 0.02
                            metrics["c"] = float(c)

                            results.append(metrics)

                if os.path.exists(os.path.join(root, dataset, c, exp, "variants")):
                    for method in os.listdir(
                        os.path.join(root, dataset, c, exp, "variants")
                    ):
                        with open(
                            os.path.join(
                                root,
                                dataset,
                                c,
                                exp,
                                "variants",
                                method,
                                "metric_values.json",
                            ),
                            "r",
                        ) as f:
                            metrics = json.load(f)

                            metrics["Dataset"] = dataset
                            metrics["Experiment"] = exp_num
                            # metrics['c'] = 0.5 if 20 > (exp_num % 20) >= 10 else 0.02
                            metrics["c"] = float(c)

                            results.append(metrics)

                if os.path.exists(os.path.join(root, dataset, c, exp, "occ")):
                    for occ_method in os.listdir(
                        os.path.join(root, dataset, c, exp, "occ")
                    ):
                        with open(
                            os.path.join(
                                root,
                                dataset,
                                c,
                                exp,
                                "occ",
                                occ_method,
                                "metric_values.json",
                            ),
                            "r",
                        ) as f:
                            metrics = json.load(f)

                            metrics["Dataset"] = dataset
                            metrics["Experiment"] = exp_num
                            # metrics['c'] = 0.5 if 20 > (exp_num % 20) >= 10 else 0.02
                            metrics["c"] = float(c)

                            results.append(metrics)
            except:
                continue
        # break

results_df = pd.DataFrame.from_records(results)
results_df = (
    results_df.assign(
        BaseMethod=results_df.Method.str.replace(
            "\+(Storey|VAERisk|PValBootstrap|G[0-9])", "", regex=True
        )
    )
    .assign(Storey=np.where(results_df.Method.str.contains("\+Storey"), "Storey", "-"))
    .assign(
        VAERiskTraining=np.where(
            results_df.Method.str.contains("\+VAERisk"), "-", "no VAE risk training"
        )
    )
    .assign(
        Bootstrap=np.where(
            results_df.Method.str.contains("\+PValBootstrap"), "p-val", "-"
        )
    )
    .assign(
        GenerateAveraging=np.where(
            ~pd.isnull(results_df.Method.str.extract("\+G([0-9])")[0]),
            results_df.Method.str.extract("\+G([0-9])")[0],
            "-",
        )
    )
)

results_df = results_df.drop(columns="Method").rename(columns={"BaseMethod": "Method"})

results_df = results_df[~results_df.Method.str.contains("-no S info")]
results_df = results_df[
    ~(
        (results_df.Method.str.contains("OddsRatio"))
        & ~(results_df.Method.str.contains("PUprop"))
    )
]

results_df.Method = np.where(
    results_df.Method == "A^3",
    r"$A^3$",
    results_df.Method,
)
results_df.Method = np.where(
    results_df.Method == "EM",
    "SAR-EM",
    results_df.Method,
)
results_df.Method = np.where(
    results_df.Method == "No OCC",
    r"VP",
    results_df.Method,
)
results_df.Method = results_df.Method.str.replace(
    "-no S info", " -no S info", regex=False
)
results_df.Method = results_df.Method.str.replace("-e100-lr1e-4-ES", "", regex=False)
results_df.Method = results_df.Method.str.replace(" +S rule", "+S", regex=False)
results_df.Method = results_df.Method.str.replace("SRuleOnly", "VP", regex=False)
results_df.Method = results_df.Method.str.replace(
    "OddsRatio-PUprop", "VP-B", regex=False
)

results_df.Dataset = results_df.Dataset.str.replace(
    "Synthetic (X, S) - logistic-interceptonly^10", "Synth. 2", regex=False
)
results_df.Dataset = results_df.Dataset.str.replace(
    "Synthetic (X, S) - logistic-interceptonly", "Synth. 1", regex=False
)
results_df.Dataset = results_df.Dataset.str.replace(
    "Synthetic (X, S) - 1-2-diagonal", "Synth. 3", regex=False
)
results_df.Dataset = results_df.Dataset.str.replace(
    "Synthetic (X, S) - SCAR", "Synth. SCAR", regex=False
)
results_df.Dataset = results_df.Dataset.str.replace("^10", "$^{10}$", regex=False)
results_df.Dataset = results_df.Dataset.str.replace("CarTruck", "CT", regex=False)
results_df.Dataset = results_df.Dataset.str.replace("MachineAnimal", "MA", regex=False)
results_df


def process_results(
    df_name,
    min_exp,
    max_exp,
    methods_filter,
    dataset_filter,
    grouping_cols,
    result_cols,
    multicolumn=False,
    plot_results=True,
    scaling=0.9,
    table_position="htbp",
):
    filtered_df = results_df

    for dataset, name in [
        ("CIFAR_CarTruck_red_val", "CIFAR CarTruck"),
        ("CIFAR_MachineAnimal_red_val", "CIFAR MachineAnimal"),
        ("STL_MachineAnimal_red_val", "STL MachineAnimal"),
        ("MNIST_35_bold_val", "MNIST 3v5"),
        ("MNIST_evenodd_bold_val", "MNIST OvE"),
        ("gas-concentrations", "Gas Concentrations"),
        ("STL_MachineAnimal_val", "STL MachineAnimal SCAR"),
    ]:
        filtered_df.Dataset = np.where(
            filtered_df.Dataset == dataset, name, filtered_df.Dataset
        )

    if min_exp is not None:
        filtered_df = filtered_df.loc[filtered_df.Experiment >= min_exp]
    if max_exp is not None:
        filtered_df = filtered_df.loc[filtered_df.Experiment < max_exp]
    if methods_filter is not None:
        filtered_df = filtered_df.loc[np.isin(filtered_df.Method, methods_filter)]
    if dataset_filter is not None:
        filtered_df = filtered_df.loc[filtered_df.Dataset == dataset_filter]

    processed_results = filtered_df.drop(columns="Experiment")
    if methods_filter is not None:
        if "Vanilla VAE-PU" in methods_filter and "Method" in grouping_cols:
            processed_results["IsNotBaseline"] = ~(
                processed_results.Method.str.contains("Vanilla VAE-PU")
                | processed_results.Method.str.contains("SAR-EM")
                | processed_results.Method.str.contains("LBE")
                | processed_results.Method.str.contains("2-step")
                | processed_results.Method.str.contains("EM-PU")
                | processed_results.Method.str.contains("MixupPU")
            )
            grouping_cols_copy = grouping_cols
            grouping_cols_copy.insert(
                grouping_cols_copy.index("Method"), "IsNotBaseline"
            )

            processed_results = processed_results.sort_values(grouping_cols_copy)

    processed_results_mean = (
        processed_results.groupby(grouping_cols).mean().round(4) * 100
    )
    processed_results_sem = (
        processed_results.groupby(grouping_cols).sem().round(4) * 100
    )
    processed_results_counts = processed_results.groupby(grouping_cols).size()
    display(processed_results_counts)

    if "IsNotBaseline" in processed_results_mean.index.names:
        processed_results_mean.index = processed_results_mean.index.droplevel(
            "IsNotBaseline"
        )
        processed_results_sem.index = processed_results_sem.index.droplevel(
            "IsNotBaseline"
        )

    if result_cols is not None:
        processed_results_mean = processed_results_mean.loc[:, result_cols]
        processed_results_sem = processed_results_sem.loc[:, result_cols]

    os.makedirs(os.path.join("processed_results", df_name), exist_ok=True)
    os.makedirs(os.path.join("processed_results", "_all_tables"), exist_ok=True)
    os.makedirs(os.path.join("processed_results", "_all_plots"), exist_ok=True)
    processed_results_mean.to_csv(
        os.path.join("processed_results", df_name, "metrics.csv")
    )

    # PLOT RESULTS

    if plot_results:
        for metric in [
            "Accuracy",
            "F1 score",
            "Precision",
            "Recall",
            "U-Accuracy",
            "U-F1 score",
            "U-Precision",
            "U-Recall",
        ]:
            # sns.set_theme()
            sns.set_style("whitegrid")
            plt.figure(figsize=(12, 8))
            ax = plt.gca()
            ax.tick_params(color=".3", labelcolor=".3")
            ax.xaxis.label.set_color(".3")
            ax.yaxis.label.set_color(".3")
            ax.spines[["top", "right", "bottom", "left"]].set_color(".3")

            plot_df = processed_results.reset_index(drop=False)
            plot_df[
                [
                    "Accuracy",
                    "F1 score",
                    "Precision",
                    "Recall",
                    "U-Accuracy",
                    "U-F1 score",
                    "U-Precision",
                    "U-Recall",
                ]
            ] *= 100
            plot_df["Style"] = np.where(
                (
                    plot_df.Method.str.contains("\+S")
                    | plot_df.Method.str.contains("S-Prophet")
                ),
                "S rule",
                "No S rule",
                # np.where(
                #     (plot_df.Method.str.contains("-no S info")),
                #     "Pure Y rule",
                #     "Y rule + S=1",
                # ),
            )
            style_order = ["No S rule", "S rule"]
            # style_order = ["Y rule + S=1", "S rule", "Pure Y rule"]

            plot_df["Hue"] = plot_df.Method.str.replace(
                " \+S rule", "", regex=True
            ).str.replace(" -no S info", "", regex=True)

            sns.lineplot(
                data=plot_df,
                x="c",
                y=metric,
                hue="Hue",
                style="Style",
                style_order=style_order,
                # err_style="bars",
                ci=68,
                # err_kws={"capsize": 3},
                palette={
                    "VP": "gray",
                    "VP+S": "gray",
                    "Vanilla VAE-PU -no S info": "gray",
                    "LBE": "#FCD542",
                    "LBE+S": "#FCD542",
                    "LBE -no S info": "#FCD542",
                    "Y-Prophet": "green",
                    "S-Prophet": "blue",
                    "Y-Prophet -no S info": "green",
                    "VAE-PU+OddsRatio": "red",
                    "VAE-PU+OddsRatio +S rule": "red",
                    "VAE-PU+OddsRatio -no S info": "red",
                    "VP-B": "#670089",
                    "VP-B+S": "#670089",
                    "VAE-PU+OddsRatio-PUprop -no S info": "#670089",
                },
                markers=True,
                markeredgewidth=0.5,
                markeredgecolor=("0.95"),
                alpha=0.8,
            )
            # sns.boxplot(data=plot_df, x='c', y='Accuracy', hue='Method')
            # sns.barplot(data=plot_df, x='c', y='Accuracy', hue='Method')
            # , err_style='bars', ci=68, err_kws={'capsize': 3})
            plt.xlabel("Label frequency $c$")
            plt.ylabel(f"{metric} [%]")
            plt.xlim(0, 0.92)
            plt.ylim(None, 100)

            plt.savefig(
                os.path.join("processed_results", df_name, f"{metric}.png"),
                dpi=300,
                bbox_inches="tight",
                transparent=True,
            )
            plt.savefig(
                os.path.join(
                    "processed_results",
                    "_all_plots",
                    f'{metric}-{df_name.replace(" ", "_")}.png',
                ),
                dpi=300,
                bbox_inches="tight",
                transparent=True,
            )
            plt.savefig(
                os.path.join(
                    "processed_results",
                    "_all_plots",
                    f'{metric}-{df_name.replace(" ", "_")}.pdf',
                ),
                bbox_inches="tight",
                transparent=True,
            )
            if metric == "Accuracy":
                plt.show()
            plt.close()

    # PREPARE RESULT TABLES

    def highlight_max(df, value_df):
        is_max = value_df.groupby(level=0).transform("max").eq(value_df)

        # max_df = pd.DataFrame(df, index=df.index, columns=df.columns)
        # max_df = max_df.applymap(lambda a: f'{a:.2f}')
        max_df = pd.DataFrame(
            np.where(is_max == True, "\\textbf{" + df + "}", df),
            index=df.index,
            columns=df.columns,
        )
        return max_df

    processed_results = (
        processed_results_mean.applymap(lambda a: f"{a:.2f}")
        + " $\pm$ "
        + processed_results_sem.applymap(lambda a: f"{a:.2f}")
    )
    processed_results = highlight_max(processed_results, processed_results_mean)

    include_caption = True
    include_label = True

    latex_table = processed_results.to_latex(
        index=True,
        escape=False,
        multirow=True,
        caption=df_name + "." if include_caption else None,
        label="tab:" + df_name.replace(" ", "_") if include_label else None,
        position=(
            None
            if not include_label and not include_caption
            else "tbp" if table_position is None else table_position
        ),
    )
    cline_start = len(processed_results.index.names)
    cline_end = cline_start + len(processed_results.columns)

    # add full rule before baseline
    # latex_table = re.sub(r'(\\\\.*?\n)(.*?)Vanilla VAE-PU', r'\1\\midrule \n\2Vanilla VAE-PU', latex_table)

    # add mid rule after LBE or EM
    # latex_table = re.sub(r'(LBE.*? \\\\)', r'\1 \\cline{' \
    #     + str(cline_start) + '-' + str(cline_end) + \
    # '}', latex_table)
    latex_table = re.sub(
        r"(SAR-EM.*? \\\\)",
        r"\1 \\cline{" + str(cline_start) + "-" + str(cline_end) + "}",
        latex_table,
    )
    # latex_table = re.sub(r'(EM.*? \\\\)', r'\1 \\cline{' \
    #     + str(cline_start) + '-' + str(cline_end) + \
    # '}', latex_table)
    # latex_table = re.sub(r'(Vanilla VAE-PU.*? \\\\)', r'\1 \\cmidrule{' \
    #     + str(cline_start) + '-' + str(cline_end) + \
    # '}', latex_table)

    # merge headers
    def merge_headers(latex_table):
        table_lines = latex_table.split("\n")
        tabular_start = 0
        tabular_end = len(table_lines) - 3

        if include_caption or include_label:
            tabular_start += 3
            tabular_end -= 1
        if include_caption and include_label:
            tabular_start += 1

        def process_line(l):
            return [
                "\\textbf{" + name.replace("\\", "").strip() + "}"
                for name in l.split("&")
                if name.replace("\\", "").strip() != ""
            ]

        header_line, index_line = (
            table_lines[tabular_start + 2],
            table_lines[tabular_start + 3],
        )
        headers = process_line(header_line)
        index_names = process_line(index_line)

        new_headers = index_names + headers
        new_headers[-1] += " \\\\"
        new_headers = " & ".join(new_headers)

        table_lines.remove(header_line)
        table_lines.remove(index_line)
        table_lines.insert(tabular_start + 2, new_headers)

        table_lines = [
            "\t" + l if i > tabular_start and i < tabular_end else l
            for i, l in enumerate(table_lines)
        ]
        if include_caption or include_label:
            table_start = 0
            table_end = len(table_lines) - 2
            table_lines = [
                "\t" + l if i > table_start and i < table_end else l
                for i, l in enumerate(table_lines)
            ]

        # insert scaling
        table_lines.insert(tabular_end + 1, "}")
        table_lines.insert(tabular_start, "\scalebox{" + f"{scaling:.2f}" + "}{")
        # insert scaling

        return "\n".join(table_lines)

    latex_table = merge_headers(latex_table)

    if multicolumn:
        latex_table = latex_table.replace("{table}", "{table*}")

    with open(os.path.join("processed_results", df_name, "metrics.tex"), "w") as f:
        f.write(latex_table)
    with open(
        os.path.join(
            "processed_results", "_all_tables", f'{df_name.replace(" ", "_")}.tex'
        ),
        "w",
    ) as f:
        f.write(latex_table)

    print(df_name)
    display(processed_results)

    # return processed_results


# %%
for df_name, dataset_filter in [
    (
        "Synthetic (X, S) -- Logistic interceptonly -- no-SCAR",
        "Synth. 1",
    ),
    (
        "Synthetic (X, S) -- Logistic^{10} interceptonly -- no-SCAR",
        "Synth. 2",
    ),
    (
        "Synthetic (X, S) -- 1-2-diagonal -- no-SCAR",
        "Synth. 3",
    ),
    (
        "Synthetic (X, S) -- SCAR",
        "Synth. SCAR",
    ),
    (
        "Synthetic (X, S) -- Logistic interceptonly (small) -- no-SCAR",
        "Synth. 1-small",
    ),
    (
        "Synthetic (X, S) -- Logistic^{10} interceptonly (small) -- no-SCAR",
        "Synth. 2-small",
    ),
    (
        "Synthetic (X, S) -- 1-2-diagonal (small) -- no-SCAR",
        "Synth. 3-small",
    ),
    (
        "Synthetic (X, S) -- SCAR (small)",
        "Synth. SCAR-small",
    ),
    (
        "MNIST 3v5 -- no-SCAR",
        "MNIST 3v5",
    ),
    (
        "CIFAR CarTruck -- no-SCAR",
        "CIFAR CT",
    ),
    (
        "STL MachineAnimal -- no-SCAR",
        "STL MA",
    ),
    (
        "Gas Concentrations -- no-SCAR",
        "Gas Concentrations",
    ),
    (
        "CDC Diabetes -- no-SCAR",
        "CDC-Diabetes",
    ),
    (
        "MNIST OvE -- no-SCAR",
        "MNIST OvE",
    ),
    (
        "CIFAR MachineAnimal -- no-SCAR",
        "CIFAR MA",
    ),
]:
    min_exp, max_exp = 0, 10
    methods_filter = [
        "S-Prophet",
        "Y-Prophet",
        "VP",
        "VP+S",
        "VP-B",
        "VP-B+S",
        "LBE",
        "LBE+S",
        # "Y-Prophet",
        # "LBE",
        # "Vanilla VAE-PU",
        # "VAE-PU+OddsRatio",
        # "VAE-PU+OddsRatio-PUprop",
        # "S-Prophet",
        # "LBE +S rule",
        # "Vanilla VAE-PU +S rule",
        # "VAE-PU+OddsRatio +S rule",
        # "VAE-PU+OddsRatio-PUprop +S rule",
        # # "Y-Prophet -no S info",
        # # "LBE -no S info",
        # # "Vanilla VAE-PU -no S info",
        # # "VAE-PU+OddsRatio -no S info",
        # # "VAE-PU+OddsRatio-PUprop -no S info",
        # "OC-SVM",
        # "IsolationForest",
        # "ECODv2",
        # r"$A^3$",
        # "OddsRatio-e10-lr1e-4",
        # "OddsRatio-e10-lr1e-4 +S rule",
        # "OddsRatio-PUprop-e10-lr1e-4",
        # "OddsRatio-PUprop-e10-lr1e-4 +S rule",
        # "SRuleOnly-e10-lr1e-4 +S rule",
        # # "OddsRatio-e100-lr1e-3",
        # # "OddsRatio-e200-lr1e-4",
        # # "OddsRatio-e100-lr1e-3 +S rule",
        # # "OddsRatio-e200-lr1e-4 +S rule",
        # # "OddsRatio-PUprop-e100-lr1e-3",
        # # "OddsRatio-PUprop-e200-lr1e-4",
        # # "OddsRatio-PUprop-e100-lr1e-3 +S rule",
        # # "OddsRatio-PUprop-e200-lr1e-4 +S rule",
        # # "SRuleOnly-e100-lr1e-3 +S rule",
        # # "SRuleOnly-e200-lr1e-4 +S rule",
        # "SAR-EM",
    ]
    grouping_cols = ["c", "Method"]
    result_cols = ["Accuracy", "U-Accuracy", "Precision", "Recall", "F1 score"]
    multicolumn = True
    # result_cols = ['Accuracy', 'F1 score']

    process_results(
        df_name,
        min_exp,
        max_exp,
        methods_filter,
        dataset_filter,
        grouping_cols,
        result_cols,
        multicolumn=multicolumn,
        scaling=0.75,
    )


# %%
def process_results_v2(
    min_exp,
    max_exp,
    methods_filter,
    dataset_order,
    multicolumn=False,
    scaling=0.75,
    suffix="",
    table_name="",
    table_position="htbp",
):
    filtered_df = results_df

    for dataset, name in [
        ("CIFAR_CarTruck_red_val", "CIFAR CarTruck"),
        ("CIFAR_MachineAnimal_red_val", "CIFAR MachineAnimal"),
        ("STL_MachineAnimal_red_val", "STL MachineAnimal"),
        ("MNIST_35_bold_val", "MNIST 3v5"),
        ("MNIST_evenodd_bold_val", "MNIST OvE"),
        ("gas-concentrations", "Gas Concentrations"),
        ("STL_MachineAnimal_val", "STL MachineAnimal SCAR"),
    ]:
        filtered_df.Dataset = np.where(
            filtered_df.Dataset == dataset, name, filtered_df.Dataset
        )

    # scar_datasets = [
    #     dataset for dataset in filtered_df.Dataset.unique() if "SCAR" in dataset
    # ]
    filtered_df = filtered_df.loc[np.isin(filtered_df.Dataset, dataset_order)]

    if min_exp is not None:
        filtered_df = filtered_df.loc[filtered_df.Experiment >= min_exp]
    if max_exp is not None:
        filtered_df = filtered_df.loc[filtered_df.Experiment < max_exp]
    if methods_filter is not None:
        filtered_df = filtered_df.loc[np.isin(filtered_df.Method, methods_filter)]

    filtered_df["Method"] = pd.Categorical(filtered_df["Method"], methods_filter)
    filtered_df["Dataset"] = pd.Categorical(filtered_df["Dataset"], dataset_order)

    filtered_df["FDR"] = 1 - filtered_df["Precision"]
    filtered_df["U-FDR"] = 1 - filtered_df["U-Precision"]

    for metric in [
        *("Accuracy", "Precision", "Recall", "F1 score"),
        *("U-Accuracy", "U-Precision", "U-Recall", "U-F1 score"),
        # *("AUC", "U-AUC", "AUC-v2", "U-AUC-v2"),
        *("FDR", "U-FDR"),
    ]:
        best_function = "max"
        if "FDR" in metric:
            best_function = "min"

        processed_results = (
            filtered_df.pivot_table(
                values=metric,
                index=["c", "Method"],
                columns="Dataset",
                aggfunc=pd.DataFrame.mean,
            ).round(4)
            * 100
        )
        processed_results_sem = (
            filtered_df.pivot_table(
                values=metric,
                index=["c", "Method"],
                columns="Dataset",
                aggfunc=pd.DataFrame.sem,
            ).round(4)
            * 100
        )

        if len(processed_results_sem.index) != len(processed_results.index) or len(
            processed_results_sem.columns
        ) != len(processed_results.columns):
            if len(processed_results_sem) == 0:
                processed_results_sem = pd.DataFrame()
            processed_results_sem = processed_results_sem.reindex_like(
                processed_results
            )

        os.makedirs(os.path.join("processed_results", "Metrics"), exist_ok=True)
        os.makedirs(os.path.join("processed_results", "_all_tables"), exist_ok=True)
        processed_results.to_csv(
            os.path.join("processed_results", "Metrics", f"{metric}-{suffix}.csv")
        )

        # # PREPARE RESULT TABLES

        processed_results.columns.name = None

        def highlight_best(df, value_df, best_function="max"):
            prophets = [
                "Prophet" in i[1] or "true s(x)" in i[1] or "true y(x)" in i[1]
                for i in value_df.index
            ]
            non_prophets = [
                "Prophet" not in i[1]
                and "true s(x)" not in i[1]
                and "true y(x)" not in i[1]
                for i in value_df.index
            ]

            is_best = (
                value_df[non_prophets].groupby(level=0).transform(best_function)
            ).eq(value_df[non_prophets])
            if np.any(prophets):
                is_best = pd.concat(
                    [
                        value_df[prophets].groupby(level=0).transform(lambda x: False),
                        is_best,
                    ],
                )

            is_best = is_best.loc[value_df.index]

            # max_df = pd.DataFrame(df, index=df.index, columns=df.columns)
            # max_df = max_df.applymap(lambda a: f'{a:.2f}')
            max_df = pd.DataFrame(
                np.where(is_best == True, "\\textbf{" + df + "}", df),
                index=is_best.index,
                columns=is_best.columns,
            )
            return max_df

        # display(processed_results.index)
        display(processed_results)

        processed_results_text = (
            processed_results.applymap(lambda a: f"{a:.2f}")
            + " $\pm$ "
            + processed_results_sem.applymap(lambda a: f"{a:.2f}")
        )
        processed_results = highlight_best(
            processed_results_text, processed_results, best_function
        )

        include_caption = True
        include_label = True

        latex_table = processed_results.to_latex(
            index=True,
            escape=False,
            multirow=True,
            caption=f"{metric} values -- {table_name}" if include_caption else None,
            label=(
                "tab:" + metric.replace(" ", "_") + f"-{suffix}"
                if include_label
                else None
            ),
            position=(
                None
                if not include_label and not include_caption
                else ("tbp" if table_position is not None else table_position)
            ),
        )
        cline_start = len(processed_results.index.names)
        cline_end = cline_start + len(processed_results.columns)

        # add full rule before baseline
        # latex_table = re.sub(r'(\\\\.*?\n)(.*?)Baseline', r'\1\\midrule \n\2Baseline', latex_table)

        # add mid rule after LBE or EM
        # latex_table = re.sub(r'(LBE.*? \\\\)', r'\1 \\cline{' \
        #     + str(cline_start) + '-' + str(cline_end) + \
        # '}', latex_table)
        latex_table = re.sub(
            r"(SAR-EM.*? \\\\)",
            r"\1 \\cline{" + str(cline_start) + "-" + str(cline_end) + "}",
            latex_table,
        )
        # latex_table = re.sub(r'(EM.*? \\\\)', r'\1 \\cline{' \
        #     + str(cline_start) + '-' + str(cline_end) + \
        # '}', latex_table)
        # latex_table = re.sub(r'(Baseline.*? \\\\)', r'\1 \\cmidrule{' \
        #     + str(cline_start) + '-' + str(cline_end) + \
        # '}', latex_table)

        # merge headers
        def merge_headers(latex_table):
            table_lines = latex_table.split("\n")
            tabular_start = 0
            tabular_end = len(table_lines) - 3

            if include_caption or include_label:
                tabular_start += 3
                tabular_end -= 1
            if include_caption and include_label:
                tabular_start += 1

            def process_line(l):
                return [
                    "\\textbf{" + name.replace("\\", "").strip() + "}"
                    for name in l.split("&")
                    if name.replace("\\", "").strip() != ""
                ]

            header_line, index_line = (
                table_lines[tabular_start + 2],
                table_lines[tabular_start + 3],
            )
            headers = process_line(header_line)
            index_names = process_line(index_line)

            new_headers = index_names + headers
            new_headers[-1] += " \\\\"
            new_headers = " & ".join(new_headers)

            table_lines.remove(header_line)
            table_lines.remove(index_line)
            table_lines.insert(tabular_start + 2, new_headers)

            table_lines = [
                "\t" + l if i > tabular_start and i < tabular_end else l
                for i, l in enumerate(table_lines)
            ]
            if include_caption or include_label:
                table_start = 0
                table_end = len(table_lines) - 2
                table_lines = [
                    "\t" + l if i > table_start and i < table_end else l
                    for i, l in enumerate(table_lines)
                ]

            # insert scaling
            table_lines.insert(tabular_end + 1, "}")
            table_lines.insert(tabular_start, "\scalebox{" + f"{scaling:.2f}" + "}{")
            # insert scaling

            return "\n".join(table_lines)

        latex_table = merge_headers(latex_table)

        if multicolumn:
            latex_table = latex_table.replace("{table}", "{table*}")
        latex_table = latex_table.replace(
            "\\centering",
            "\\centering \\scriptsize \\renewcommand{\\arraystretch}{1.2}",
        )

        with open(
            os.path.join("processed_results", "Metrics", f"{metric}-{suffix}.tex"), "w"
        ) as f:
            f.write(latex_table)
        with open(
            os.path.join("processed_results", "_all_tables", f"{metric}-{suffix}.tex"),
            "w",
        ) as f:
            f.write(latex_table)

        print(f"{metric} df")
        display(processed_results)


### ---------------------------------------------------------

min_exp, max_exp = 0, 10

methods_filter = [
    # "S-Prophet",
    "VP+S",
    "VP-B+S",
    # "VP-B+S + true s(x)",
    # "VP-B+S + true y(x)",
    "LBE+S",
]
dataset_order = [
    "Synth. 1",
    "Synth. 2",
    "Synth. 3",
    "Synth. SCAR",
]

process_results_v2(
    min_exp,
    max_exp,
    methods_filter,
    dataset_order,
    multicolumn=False,
    scaling=0.8,
    suffix="Synth+S",
    table_name="Method comparison -- Synthetic datasets",
)

# ----------------

methods_filter = [
    "VP+S",
    "VP-B+S",
    "LBE+S",
]
dataset_order = [
    "MNIST 3v5",
    "MNIST OvE",
    "CIFAR CT",
    "CIFAR MA",
    "STL MA",
    "CDC-Diabetes",
]

process_results_v2(
    min_exp,
    max_exp,
    methods_filter,
    dataset_order,
    multicolumn=False,
    scaling=0.63,
    suffix="Real+S",
    table_name="Method comparison -- Real-world datasets",
)

# ----------------

methods_filter = [
    "S-Prophet",
    "Y-Prophet",
    # "VP",
    # "VP+S",
    "VP-B",
    "VP-B+S",
    # "LBE",
    # "LBE+S",
    "VP-B+S + true s(x)",
    "VP-B+S + true y(x)",
]
dataset_order = [
    "Synth. 1",
    "Synth. 2",
    "Synth. 3",
    "Synth. SCAR",
]

process_results_v2(
    min_exp,
    max_exp,
    methods_filter,
    dataset_order,
    multicolumn=False,
    scaling=0.8,
    suffix="Synth-Compare",
    table_name="Decision rule comparison -- Synthetic datasets",
)

# ----------------

methods_filter = [
    "S-Prophet",
    "Y-Prophet",
    # "VP",
    # "VP+S",
    "VP-B",
    "VP-B+S",
    # "LBE",
    # "LBE+S",
]
dataset_order = [
    "MNIST 3v5",
    "MNIST OvE",
    "CIFAR CT",
    "CIFAR MA",
    "STL MA",
    "CDC-Diabetes",
]

process_results_v2(
    min_exp,
    max_exp,
    methods_filter,
    dataset_order,
    multicolumn=False,
    scaling=0.63,
    suffix="Real-Compare",
    table_name="Decision rule comparison -- Real-world datasets",
)

# # %%

# # ==================================================

# methods_filter = [
#     "VP+S",
#     "VP-B+S",
#     "LBE+S",
# ]
# dataset_order = [
#     "MNIST 3v5 - Synthetic InterceptOnly",
#     "MNIST OvE - Synthetic InterceptOnly",
#     "CIFAR CT - Synthetic InterceptOnly",
#     "CIFAR MA - Synthetic InterceptOnly",
#     "STL MA - Synthetic InterceptOnly",
#     "CDC-Diabetes - Synthetic InterceptOnly",
# ]

# process_results_v2(
#     min_exp,
#     max_exp,
#     methods_filter,
#     dataset_order,
#     multicolumn=False,
#     scaling=0.63,
#     suffix="Real-Synth+S",
#     table_name="Method comparison -- Real-world datasets (synthetic labeling)",
# )

# # ----------------

# methods_filter = [
#     "S-Prophet",
#     "Y-Prophet",
#     # "VP",
#     # "VP+S",
#     "VP-B",
#     "VP-B+S",
#     # "LBE",
#     # "LBE+S",
# ]
# dataset_order = [
#     "MNIST 3v5 - Synthetic InterceptOnly",
#     "MNIST OvE - Synthetic InterceptOnly",
#     "CIFAR CT - Synthetic InterceptOnly",
#     "CIFAR MA - Synthetic InterceptOnly",
#     "STL MA - Synthetic InterceptOnly",
#     "CDC-Diabetes - Synthetic InterceptOnly",
# ]

# process_results_v2(
#     min_exp,
#     max_exp,
#     methods_filter,
#     dataset_order,
#     multicolumn=False,
#     scaling=0.63,
#     suffix="Real-Synth-Compare",
#     table_name="Decision rule comparison -- Real-world datasets (synthetic labeling)",
# )


# # %%

# # ==================================================

# methods_filter = [
#     "VP+S",
#     "VP-B+S",
#     "LBE+S",
# ]
# dataset_order = [
#     "MNIST 3v5 - proba sampling",
#     "MNIST OvE - proba sampling",
#     "CIFAR CT - proba sampling",
#     "CIFAR MA - proba sampling",
#     "STL MA - proba sampling",
#     "CDC-Diabetes",
# ]

# process_results_v2(
#     min_exp,
#     max_exp,
#     methods_filter,
#     dataset_order,
#     multicolumn=False,
#     scaling=0.63,
#     suffix="Real-Proba+S",
#     table_name="Method comparison -- Real-world datasets (probabilistic sampling labeling)",
# )

# # ----------------

# methods_filter = [
#     "S-Prophet",
#     "Y-Prophet",
#     # "VP",
#     # "VP+S",
#     "VP-B",
#     "VP-B+S",
#     # "LBE",
#     # "LBE+S",
# ]
# dataset_order = [
#     "MNIST 3v5 - proba sampling",
#     "MNIST OvE - proba sampling",
#     "CIFAR CT - proba sampling",
#     "CIFAR MA - proba sampling",
#     "STL MA - proba sampling",
#     "CDC-Diabetes",
# ]

# process_results_v2(
#     min_exp,
#     max_exp,
#     methods_filter,
#     dataset_order,
#     multicolumn=False,
#     scaling=0.63,
#     suffix="Real-Proba-Compare",
#     table_name="Decision rule comparison -- Real-world datasets (probabilistic sampling labeling)",
# )

# # %%
# # ==================================

# methods_filter = [
#     "VP+S",
#     "VP-B+S",
#     "VP-B+S + true s(x)",
#     "VP-B+S + true y(x)",
#     "LBE+S",
# ]
# dataset_order = [
#     "Synthetic (X, S) - logistic-v2",
#     "Synthetic (X, S) - logistic$^{10}$-v2",
#     "Synthetic (X, S) - logistic-inverse-v2",
#     "Synthetic (X, S) - logistic-inverse$^{10}$-v2",
# ]

# process_results_v2(
#     min_exp,
#     max_exp,
#     methods_filter,
#     dataset_order,
#     multicolumn=False,
#     scaling=0.63,
#     suffix="Synth-v2+S",
#     table_name="Method comparison -- Synthetic datasets (v2)",
# )

# # ----------------

# methods_filter = [
#     "S-Prophet",
#     "Y-Prophet",
#     # "VP",
#     # "VP+S",
#     "VP-B",
#     "VP-B+S",
#     "VP-B+S + true s(x)",
#     "VP-B+S + true y(x)",
#     # "LBE",
#     # "LBE+S",
# ]
# dataset_order = [
#     "Synthetic (X, S) - logistic-v2",
#     "Synthetic (X, S) - logistic$^{10}$-v2",
#     "Synthetic (X, S) - logistic-inverse-v2",
#     "Synthetic (X, S) - logistic-inverse$^{10}$-v2",
# ]

# process_results_v2(
#     min_exp,
#     max_exp,
#     methods_filter,
#     dataset_order,
#     multicolumn=False,
#     scaling=0.63,
#     suffix="Synth-v2-Compare",
#     table_name="Decision rule comparison -- Synthetic datasets (v2)",
# )

# # %%
