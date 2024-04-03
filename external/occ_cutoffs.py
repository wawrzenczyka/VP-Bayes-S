import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from joblib import Parallel, delayed
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit


def select_indices(idx, *args):
    return tuple([x[idx] for x in args])


class Cutoff:
    is_fitted = False

    @property
    def cutoff_type(self):
        raise NotImplementedError("Use derived class for cutoff")

    def _clf_train(self, clf, X_train, y_train=None):
        return clf.fit(X_train, y_train)

    def _clf_predict(self, clf, X_test):
        scores = clf.score_samples(X_test)
        return scores

    def fit(self, X_train, y_train=None):
        raise NotImplementedError("Use derived class for cutoff")

    def apply(self, X_test, inlier_rate):
        raise NotImplementedError("Use derived class for cutoff")

    def fit_apply(self, X_train, X_test, inlier_rate, y_train=None):
        self.fit(X_train, y_train)
        return self.apply(X_test, inlier_rate)

    def get_clfs(self):
        raise NotImplementedError("Use derived class for cutoff")

    def set_clfs(self, clfs):
        raise NotImplementedError("Use derived class for cutoff")


class EmpiricalCutoff(Cutoff):
    cutoff_type = "Empirical"

    def __init__(self, construct_clf):
        self.clf = construct_clf()

    def fit(self, X_train, y_train=None):
        self._clf_train(self.clf, X_train, y_train)
        self.is_fitted = True

    def apply(self, X_test, inlier_rate):
        scores = self._clf_predict(self.clf, X_test)
        # self.threshold_ = np.quantile(scores, q=1 - inlier_rate)
        # y_pred = np.where(scores > self.threshold_, 1, 0)

        pval_order = np.argsort(scores)[::-1]
        n_inliers = int(inlier_rate * len(X_test))
        predicted_inliers = pval_order[:n_inliers]
        y_pred = np.where(np.isin(np.arange(len(X_test)), predicted_inliers), 1, 0)

        return scores, y_pred

    def get_clfs(self):
        return [self.clf]

    def set_clfs(self, clfs):
        self.clf = clfs[0]


import contextlib


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class MultisplitCutoff(Cutoff):
    @property
    def cutoff_type(self):
        return (
            "Multisplit"
            + (
                f"-{self.median_multiplier}_median"
                if self.median_multiplier != 1
                else ""
            )
            + (
                f"-{self.resampling_repeats}_repeat"
                if self.resampling_repeats != 10
                else ""
            )
        )

    def __init__(
        self, construct_clf, alpha, resampling_repeats, median_multiplier=1, n_jobs=-1
    ):
        self.construct_clf = construct_clf
        self.clfs = []
        for _ in range(resampling_repeats):
            self.clfs.append(construct_clf())

        self.alpha = alpha
        self.resampling_repeats = resampling_repeats
        self.median_multiplier = median_multiplier
        self.n_jobs = n_jobs

    def _calculate_ith_cal_scores(self, clf, i: int, X_train, y_train=None):
        N = len(X_train)

        with temp_seed(i):
            multisplit_samples = np.random.choice(
                range(N), size=int(N / 2), replace=False
            )
            is_multisplit_sample = np.isin(range(N), multisplit_samples)
            if y_train is not None:
                for sample_class in [0, 1]:
                    if (
                        np.sum(y_train[is_multisplit_sample] == sample_class) == 0
                    ):  # no sample of one class
                        first_class_sample = np.where(y_train == sample_class)[0][0]
                        is_multisplit_sample[
                            first_class_sample
                        ] = True  # to avoid no positive/negative samples
                y_multi_train, y_multi_cal = (
                    y_train[is_multisplit_sample],
                    y_train[~is_multisplit_sample],
                )
            else:
                y_multi_train, y_multi_cal = None, None
            X_multi_train, X_multi_cal = (
                X_train[is_multisplit_sample],
                X_train[~is_multisplit_sample],
            )

            clf = self._clf_train(clf, X_multi_train, y_multi_train)
            cal_scores = self._clf_predict(clf, X_multi_cal)
            return clf, cal_scores

    def _prepare_multisplit_cal_scores(self, X_train, y_train=None):
        if self.n_jobs != 1:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._calculate_ith_cal_scores)(clf, i, X_train, y_train)
                for i, clf in zip(range(self.resampling_repeats), self.clfs)
            )
        else:
            results = []
            for i, clf in zip(range(self.resampling_repeats), self.clfs):
                results.append(self._calculate_ith_cal_scores(clf, i, X_train, y_train))

        self.clfs, cal_scores_all = zip(*results)
        self.clfs, cal_scores_all = list(self.clfs), list(cal_scores_all)

        return cal_scores_all

    def _calculate_ith_p_values(self, i: int, X_test):
        cal_scores = self.cal_scores_[i]
        test_scores = self._clf_predict(self.clfs[i], X_test)

        num_smaller_cal_scores = (test_scores >= cal_scores.reshape(-1, 1)).sum(axis=0)
        p_vals = (num_smaller_cal_scores + 1) / (len(cal_scores) + 1)
        return p_vals

    def _get_multisplit_p_values(self, X_test):
        if self.n_jobs != 1:
            p_vals_all = Parallel(n_jobs=self.n_jobs)(
                delayed(self._calculate_ith_p_values)(i, X_test)
                for i in range(self.resampling_repeats)
            )
        else:
            p_vals_all = []
            for i in range(self.resampling_repeats):
                p_vals_all.append(self._calculate_ith_p_values(i, X_test))
        p_vals_all = np.row_stack(p_vals_all)

        p_vals = self.median_multiplier * np.median(p_vals_all, axis=0)
        return p_vals

    def fit(self, X_train, y_train=None):
        self.cal_scores_ = self._prepare_multisplit_cal_scores(X_train, y_train)
        self.is_fitted = True
        return self.cal_scores_

    def get_p_vals(self, X_test):
        return self._get_multisplit_p_values(X_test)

    def apply_to_p_vals(self, p_vals):
        y_pred = np.where(p_vals < self.alpha, 0, 1)
        return y_pred

    def apply(self, X_test, inlier_rate):
        p_vals = self._get_multisplit_p_values(X_test)
        return p_vals, self.apply_to_p_vals(p_vals)

    def get_clfs(self):
        return self.clfs

    def set_clfs(self, clfs):
        self.clfs = clfs

    def visualize_lottery(
        self,
        visualization_data,
        test_case_name,
        clf_name,
        RESULTS_DIR,
        alpha=None,
        max_samples=100,
    ):
        X_train, y_train, X_test, y_test = visualization_data

        if len(X_test) > max_samples:
            # constant random state is important for consistency
            sss = StratifiedShuffleSplit(
                n_splits=1, random_state=42, train_size=min(len(X_test), max_samples)
            )
            for indices, _ in sss.split(X_test, y_test):
                # executes only once
                X_test, y_test = X_test[indices], y_test[indices]

        p_vals_all = np.zeros((self.resampling_repeats, len(X_test)))
        for i in range(self.resampling_repeats):
            cal_scores = self.cal_scores_[i]
            test_scores = self._clf_predict(self.clfs[i], X_test)

            num_smaller_cal_scores = (test_scores > cal_scores.reshape(-1, 1)).sum(
                axis=0
            )
            p_vals = (num_smaller_cal_scores + 1) / (len(cal_scores) + 1)
            p_vals_all[i, :] = p_vals

        p_vals = self.median_multiplier * np.median(p_vals_all, axis=0)

        p_val_range = p_vals_all.max(axis=0) - p_vals_all.min(axis=0)
        mean_range = np.mean(p_val_range)
        median_range = np.median(p_val_range)
        max_range = np.max(p_val_range)
        mean_start = np.mean(p_vals_all.min(axis=0))
        mean_end = np.mean(p_vals_all.max(axis=0))

        os.makedirs(os.path.join(RESULTS_DIR, test_case_name, "img"), exist_ok=True)

        data = p_vals_all[:, :max_samples]
        df = pd.DataFrame(
            {
                "p-value": data.reshape(-1),
                "Split": np.repeat(
                    range(1, self.resampling_repeats + 1), data.shape[1]
                ),
                "Sample": np.tile(range(1, data.shape[1] + 1), data.shape[0]),
                "Type": np.tile(
                    np.where(y_test[: data.shape[1]] == 1, "Inlier", "Outlier"),
                    data.shape[0],
                ),
            }
        )

        median_df = (
            df.groupby(["Sample", "Type"])["p-value"]
            .median()
            .rename("Median p-value")
            .reset_index(drop=False)
            .sort_values(["Type", "Median p-value"], ascending=[False, True])
            .assign(Order=range(data.shape[1]))
        )

        # sample_types = np.where(y_test == 1, 'Inlier', 'Outlier')
        hue_order = ["Inlier", "Outlier"]

        sns.set_theme()
        # fig, axs = plt.subplots(2, 1, figsize=(16, 6), height_ratios=(3, 1), sharex=True)
        fig = plt.figure(figsize=(16, 6))
        ax = plt.gca()

        scatter_1 = sns.scatterplot(
            data=median_df,
            x="Order",
            y="Median p-value",
            hue="Type",
            hue_order=hue_order,
            palette=[sns.color_palette()[0], sns.color_palette()[3]],
            zorder=100,
            s=12,
            edgecolor="k",
            ax=ax,
        )
        # scatter_2 = sns.scatterplot(
        #     data=median_df,
        #     x='Order',
        #     y='Median p-value',
        #     hue='Type',
        #     hue_order=hue_order,
        #     palette=[sns.color_palette()[0], sns.color_palette()[3]],
        #     zorder=100, s=12, edgecolor='k',
        #     ax=axs[1],
        # )
        scatter_1.axhline(alpha, ls="--", c="k")
        # scatter_2.axhline(alpha, ls='--', c='k')

        ax.legend(title="Median values")
        # axs[1].legend(title='Median values')

        sorted_df = df.merge(median_df, on=["Sample", "Type"]).sort_values("Order")
        sns.stripplot(data=sorted_df, x="Order", y="p-value", linewidth=0.2, ax=ax)
        # sns.stripplot(data=sorted_df, x='Order', y='p-value', linewidth=0.2, ax=axs[1])

        ax.tick_params(axis="x", labelsize=6)
        # ax.set_xticks(sorted_df.Order, sorted_df.Sample)
        # ax.set_xlabel('Sample number')
        ax.set_xticks([], [])
        ax.set_xlabel("")

        # axs[0].set_title(f'{self.cutoff_type} ({clf_name}) p-value lottery' + \
        #     f' - max range length: {max_range:.3f}, mean: {mean_range:.3f}, median: {median_range:.3f}, range: ({mean_start:.3f}-{mean_end:.3f})')
        # axs[1].set_title(f'Zoomed')

        ax.set_ylim(-0.005, 1 + 0.005)
        # axs[1].set_ylim(-0.005, alpha + 0.005)

        plt.savefig(
            os.path.join(
                RESULTS_DIR,
                test_case_name,
                "img",
                f"lottery-{clf_name}-{self.cutoff_type}.png",
            ),
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.savefig(
            os.path.join(
                RESULTS_DIR,
                test_case_name,
                "img",
                f"lottery-{clf_name}-{self.cutoff_type}.pdf",
            ),
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)

    def visualize_p_values(
        self, visualization_data, test_case_name, clf_name, RESULTS_DIR
    ):
        X_train, y_train, X_test, y_test = visualization_data

        p_vals = self.get_p_vals(X_test)
        train_p_vals = self.get_p_vals(X_train)

        inlier_idx = np.where(y_test == 1)[0]
        inlier_mask = np.isin(range(len(y_test)), inlier_idx)

        df = pd.DataFrame(
            {
                # 'Score': scores,
                "p-value": p_vals,
                "Class": np.where(inlier_mask, "Inlier", "Outlier"),
            }
        )
        train_df = pd.DataFrame(
            {
                # 'Score': train_scores,
                "p-value": train_p_vals,
                "Class": np.array(["Inlier"] * len(train_p_vals)),
            }
        )

        os.makedirs(os.path.join(RESULTS_DIR, test_case_name, "img"), exist_ok=True)

        # for metric in ['Score', 'p-value']:
        for metric in ["p-value"]:
            sns.set_theme()
            # _, axs = plt.subplots(1, 2, figsize=(14, 6),
            #     sharex=True, sharey=True)
            fig = plt.figure(figsize=(10, 8))
            ax = plt.gca()

            # sns.histplot(train_df, x=metric, hue='Class', ax=axs[0],
            #     hue_order=['Inlier'], stat='probability')
            # axs[0].set_title('Train')
            sns.histplot(
                df,
                x=metric,
                hue="Class",
                ax=ax,
                edgecolor="k",
                hue_order=["Inlier", "Outlier"],
                stat="probability",
            )
            # axs[1].set_title('Test')

            # plt.suptitle(f'{test_case_name} ({clf_name}, {self.cutoff_type}) - {metric} distribution')
            plt.savefig(
                os.path.join(
                    RESULTS_DIR,
                    test_case_name,
                    "img",
                    f"distribution-{metric}-{clf_name}-{self.cutoff_type}.png",
                ),
                dpi=600,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.savefig(
                os.path.join(
                    RESULTS_DIR,
                    test_case_name,
                    "img",
                    f"distribution-{metric}-{clf_name}-{self.cutoff_type}.pdf",
                ),
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close()

    def visualize_roc(self, visualization_data, test_case_name, clf_name, RESULTS_DIR):
        X_train, y_train, X_test, y_test = visualization_data
        p_vals = self.get_p_vals(X_test)

        inlier_idx = np.where(y_test == 1)[0]
        inlier_mask = np.isin(range(len(y_test)), inlier_idx)

        df = pd.DataFrame(
            {
                # 'Score': scores,
                "p-value": p_vals,
                "Class": np.where(inlier_mask, "Inlier", "Outlier"),
            }
        )

        os.makedirs(os.path.join(RESULTS_DIR, test_case_name, "img"), exist_ok=True)

        # for metric in ['Score', 'p-value']:
        for metric in ["p-value"]:
            # Plot ROC
            fpr, tpr, _ = metrics.roc_curve(y_test, df[metric], pos_label=1)

            sns.set_theme()
            plt.figure(figsize=(8, 6))
            plt.plot(
                fpr,
                tpr,
                color="darkorange",
                lw=2,
                label="ROC curve (area = %0.2f)" % metrics.auc(fpr, tpr),
            )
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.xlim([-0.01, 1.0])
            plt.ylim([0.0, 1.01])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{test_case_name} ({clf_name}) - {metric} ROC (for Inliers)")
            plt.legend(loc="lower right")

            plt.savefig(
                os.path.join(
                    RESULTS_DIR,
                    test_case_name,
                    "img",
                    f"ROC-{metric}-{clf_name}-{self.cutoff_type}.png",
                ),
                dpi=150,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.savefig(
                os.path.join(
                    RESULTS_DIR,
                    test_case_name,
                    "img",
                    f"ROC-{metric}-{clf_name}-{self.cutoff_type}.pdf",
                ),
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close()

    def visualize_calibration(
        self, visualization_data, test_case_name, clf_name, RESULTS_DIR
    ):
        vis_clfs = []
        for _ in range(self.resampling_repeats):
            vis_clfs.append(self.construct_clf())

        X_train, y_train, X_test, y_test = visualization_data
        N = len(X_train)

        train_df = pd.DataFrame()

        for i in range(self.resampling_repeats):
            multisplit_samples = np.random.choice(
                range(N), size=int(N / 2), replace=False
            )
            is_multisplit_sample = np.isin(range(N), multisplit_samples)
            if y_train is not None:
                for sample_class in [0, 1]:
                    if (
                        np.sum(y_train[is_multisplit_sample] == sample_class) == 0
                    ):  # no sample of one class
                        first_class_sample = np.where(y_train == sample_class)[0][0]
                        is_multisplit_sample[
                            first_class_sample
                        ] = True  # to avoid no positive/negative samples
                y_multi_train, y_multi_cal = (
                    y_train[is_multisplit_sample],
                    y_train[~is_multisplit_sample],
                )
            else:
                y_multi_train, y_multi_cal = None, None
            X_multi_train, X_multi_cal = (
                X_train[is_multisplit_sample],
                X_train[~is_multisplit_sample],
            )

            self._clf_train(vis_clfs[i], X_multi_train, y_multi_train)
            train_scores = self._clf_predict(vis_clfs[i], X_train)

            train_df = pd.concat(
                [
                    train_df,
                    pd.DataFrame(
                        {
                            "Split": np.char.add(
                                "Split ", np.repeat(i + 1, len(X_train)).astype(str)
                            ),
                            "Score": train_scores,
                            "SampleType": np.where(
                                is_multisplit_sample, "Train", "Calibration"
                            ),
                        }
                    ),
                ]
            )

        os.makedirs(os.path.join(RESULTS_DIR, test_case_name, "img"), exist_ok=True)

        vis_clf = self.construct_clf()
        self._clf_train(vis_clf, X_train, y_train)
        train_scores = self._clf_predict(vis_clf, X_train)

        train_df = pd.concat(
            [
                train_df,
                pd.DataFrame(
                    {
                        "Split": "Train",
                        "Score": train_scores,
                        "SampleType": "Train",
                    }
                ),
            ]
        )
        train_df = train_df.reset_index(drop=True)
        cal_df = train_df[
            (train_df.SampleType == "Calibration") | (train_df.Split == "Train")
        ]

        sns.set_theme()

        fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
        fig.suptitle(f"{test_case_name} ({clf_name}) - scores per calibration split")

        ax = axs[0]
        sns.stripplot(
            data=cal_df,
            x="Score",
            y="Split",
            orient="h",
            ax=ax,
            hue="SampleType",
            hue_order=["Train", "Calibration"],
            s=4,
            linewidth=0.2,
            alpha=0.7,
        )
        ax.set_title(f"Calibration set only")

        ax.get_legend().remove()

        ax = axs[1]
        sns.stripplot(
            data=train_df,
            x="Score",
            y="Split",
            orient="h",
            ax=ax,
            hue="SampleType",
            hue_order=["Train", "Calibration"],
            s=4,
            linewidth=0.2,
            alpha=0.7,
        )
        ax.set_title(
            f"{test_case_name} ({clf_name}) - scores per calibration split, whole train dataset"
        )
        ax.set_title(f"Whole train")

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center")
        ax.get_legend().remove()

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                RESULTS_DIR,
                test_case_name,
                "img",
                f"diagnostic-scores-{clf_name}-{self.cutoff_type}.png",
            ),
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.savefig(
            os.path.join(
                RESULTS_DIR,
                test_case_name,
                "img",
                f"diagnostic-scores-{clf_name}-{self.cutoff_type}.pdf",
            ),
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)


# --------------------------------- p-value cutoffs ---------------------------------


class PValueCutoff:
    base_cutoff: Cutoff
    is_fitted = False

    @property
    def cutoff_type(self):
        raise NotImplementedError("Use derived class for cutoff")

    @property
    def full_cutoff_type(self):
        return self.base_cutoff.cutoff_type + "+" + self.cutoff_type

    @property
    def short_cutoff_type(self):
        raise NotImplementedError("Use derived class for cutoff")

    def __init__(self, base_cutoff):
        # base cutoff needs to be fitted already
        self.base_cutoff = base_cutoff
        assert hasattr(self.base_cutoff, "get_p_vals")
        assert hasattr(self.base_cutoff, "is_fitted") and self.base_cutoff.is_fitted

    def fit_apply(self, X_test):
        self.fit(X_test)
        return self.apply(X_test)

    def fit(self, X_test):
        p_vals = self.base_cutoff.get_p_vals(X_test)

        p_vals = np.sort(p_vals)
        minimal_thresholds = self._get_minimal_thresholds(len(p_vals))
        self.threshold_ = self._calculate_final_threshold(p_vals, minimal_thresholds)

        self.is_fitted = True
        return self.threshold_

    def apply(self, X_test):
        p_vals = self.base_cutoff.get_p_vals(X_test)
        y_pred = self.apply_to_pvals(p_vals)
        return p_vals, y_pred

    def apply_to_pvals(self, p_vals):
        y_pred = np.where(p_vals <= self.threshold_, 0, 1)
        return y_pred

    def _get_thresholds(self, N):
        raise NotImplementedError("Use derived class for cutoff")

    def _get_threshold_names(self):
        raise NotImplementedError("Use derived class for cutoff")

    def _get_minimal_thresholds(self, N):
        minimal_threshold = np.min(  # Use the most tight threshold point
            np.concatenate([t.reshape(-1, 1) for t in self._get_thresholds(N)], axis=1),
            axis=1,
        )
        return minimal_threshold

    def _calculate_final_threshold(self, sorted_p_vals, minimal_thresholds):
        raise NotImplementedError("Use derived class for cutoff")

    def visualize(
        self,
        X_test,
        y_test,
        figure,
        test_case_name,
        clf_name,
        RESULTS_DIR,
        zoom=False,
        zoom_left=True,
        save_plot=False,
    ):
        assert self.is_fitted, "Cutoff needs to be fitted first"

        N = len(X_test)

        p_vals = self.base_cutoff.get_p_vals(X_test)
        thresholds = self._get_thresholds(N)
        minimal_thresholds = self._get_minimal_thresholds(N)

        p_val_order = np.argsort(p_vals)
        p_vals, y_test = p_vals[p_val_order], y_test[p_val_order]  # sort

        y_pred = self.apply_to_pvals(p_vals)

        x = np.array(range(N)) / N
        outlier_vec = np.where(y_test == 0, "Outlier", "Inlier")
        rejected_vec = np.where(y_pred == 0, "Rejected", "Not rejected")

        num_elements = N
        if zoom:
            num_elements = 100
            if zoom_left:
                if np.sum(y_pred == 0) < len(y_pred):
                    num_elements = max(num_elements, int(1.5 * np.sum(y_pred == 0)))
                num_elements = min(N, num_elements)

                x, p_vals, outlier_vec, rejected_vec, minimal_thresholds = tuple(
                    [
                        a[:num_elements]
                        for a in [
                            x,
                            p_vals,
                            outlier_vec,
                            rejected_vec,
                            minimal_thresholds,
                        ]
                    ]
                )
                thresholds = [a[:num_elements] for a in thresholds]
            else:
                if np.sum(y_pred == 0) < len(y_pred):
                    num_elements = max(num_elements, int(1.5 * np.sum(y_pred == 1)))
                num_elements = min(N, num_elements)

                x, p_vals, outlier_vec, rejected_vec, minimal_thresholds = tuple(
                    [
                        a[-num_elements:]
                        for a in [
                            x,
                            p_vals,
                            outlier_vec,
                            rejected_vec,
                            minimal_thresholds,
                        ]
                    ]
                )
                thresholds = [a[-num_elements:] for a in thresholds]

        fig, ax = figure
        sns.scatterplot(
            x=x,
            y=p_vals,
            hue=outlier_vec,
            style=rejected_vec,
            hue_order=["Inlier", "Outlier"],
            style_order=["Not rejected", "Rejected"],
            edgecolor="k",
            linewidth=0.01,
            ax=ax,
        )

        sns.scatterplot(
            x=x,
            y=minimal_thresholds,
            hue=["Threshold"] * num_elements,
            palette=["r"],
            edgecolor=None,
            s=5,
            alpha=0.7,
            zorder=10,
            ax=ax,
        )
        num_threshold_markers = 1

        if len(thresholds) > 1:
            threshold_names = self._get_threshold_names()
            threshold_palette = sns.color_palette("Greys", len(thresholds))

            for i in range(len(thresholds)):
                num_threshold_markers += 1

                sns.scatterplot(
                    x=x,
                    y=thresholds[i],
                    hue=[threshold_names[i]] * num_elements,
                    palette=[threshold_palette[i]],
                    edgecolor=None,
                    s=1.25,
                    alpha=0.8,
                    zorder=20,
                    ax=ax,
                )

        ax.set_xlabel("p-value quantile")
        ax.set_ylabel("Value")
        ax.set_ylim(-0.01, 1.01)

        legend = ax.legend()
        # resize threshold markers
        legend.legendHandles[-num_threshold_markers]._sizes = [8.0]
        if len(thresholds) > 1:
            for h in legend.legendHandles[-num_threshold_markers + 1 :]:
                h._sizes = [4.0]

        # ax.set_title(f'{self.cutoff_type}, alpha={self.alpha:.3f}' + \
        #     f'{f" (zoomed)" if zoom else ""}')

        if save_plot:
            os.makedirs(os.path.join(RESULTS_DIR, test_case_name, "img"), exist_ok=True)
            fig.tight_layout()
            base_cutoff_string = (
                "-" + self.base_cutoff.cutoff_type
                if self.base_cutoff.cutoff_type != "Multisplit"
                else ""
            )
            fig.savefig(
                os.path.join(
                    RESULTS_DIR,
                    test_case_name,
                    "img",
                    f"cutoff-{clf_name}{base_cutoff_string}-{self.short_cutoff_type}.png",
                ),
                dpi=600,
                bbox_inches="tight",
                facecolor="white",
            )
            fig.savefig(
                os.path.join(
                    RESULTS_DIR,
                    test_case_name,
                    "img",
                    f"cutoff-{clf_name}{base_cutoff_string}-{self.short_cutoff_type}.pdf",
                ),
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close(fig)


class FORControlCutoff(PValueCutoff):
    @property
    def cutoff_type(self):
        return "FOR-CTL"

    @property
    def short_cutoff_type(self):
        return "FOR-CTL"

    def __init__(self, base_cutoff, alpha, pi):
        super().__init__(base_cutoff)
        self.alpha = alpha
        self.pi = pi

    def _get_thresholds(self, N):
        i_array = np.array(range(N)) + 1  # 1, ..., N
        return [1 - ((1 - (i_array / N)) * (1 - self.alpha)) / (self.pi)]

    def _get_threshold_names(self):
        return ["FOR threshold"]

    def _calculate_final_threshold(self, sorted_p_vals, minimal_thresholds):
        # min p(i): p(i) <= threshold
        p_vals_below_bar = np.where(sorted_p_vals <= minimal_thresholds)[0]
        if len(p_vals_below_bar) > 0:
            first_fulfillment_index = p_vals_below_bar[0]
            return sorted_p_vals[first_fulfillment_index]
        else:
            return np.inf
