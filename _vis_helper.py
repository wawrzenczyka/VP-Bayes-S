# calculate_metrics

# %%

    for k in [1]:
        root_dir = f'diagnostic-plots/test/k={k:.2f}'
        file_name = 'Synthetic logistic'

        import os
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns

        os.makedirs(root_dir, exist_ok=True)

        y_pred = np.where(s == 1, 1, np.where(y_proba > (1 + s_proba) / 2, 1, 0))
        y_alt = np.where(s == 1, 1, np.where(y_proba > 0.5, 1, 0))

        sns.set_theme()
        plt.figure(figsize=(8, 5))
        idx = np.argsort(y_proba)
        plt.plot(np.arange(-len(idx), 2 * len(idx)), np.repeat(0.5, 3 * len(idx)), "k.", ms=1)
        plt.plot(np.arange(len(idx)), y_proba[idx], "b.", ms=10)

        # corrected_rule = y_proba - s_proba / 2
        corrected_rule = y_proba - s_proba / (k + 1) - (k - 1) / (2 * (k + 1))

        plt.plot(
            np.arange(len(idx))[(s[idx] == 1) & (y[idx] == 1)],
            (corrected_rule)[idx][(s[idx] == 1) & (y[idx] == 1)],
            "r.",
            ms=3,
            alpha=0.5,
        )

        plt.plot(
            np.arange(len(idx))[(s[idx] == 0) & (y[idx] == 1)],
            (corrected_rule)[idx][(s[idx] == 0) & (y[idx] == 1)],
            "g.",
            ms=3,
            alpha=0.5,
        )
        plt.plot(
            np.arange(len(idx))[(s[idx] == 0) & (y[idx] == 0)],
            (corrected_rule)[idx][(s[idx] == 0) & (y[idx] == 0)],
            "gx",
            ms=3,
            alpha=0.5,
        )

        x_classification_change = np.where(y_proba[idx] > 0.5)[0]
        plt.fill_between(np.concatenate([x_classification_change, np.arange(np.max(x_classification_change), 2 * np.max(x_classification_change))]), -1, 0.5, color='#FFC310', alpha=0.5)

        plt.legend(
            [
                "_nolegend_",
                "$y(x)$",
                "$y(x) - \\frac{s(x)}{2}$ ($S = 1$)",
                "$y(x) - \\frac{s(x)}{2}$ ($S = 0, Y = 1$)",
                "$y(x) - \\frac{s(x)}{2}$ ($S = 0, Y = -1$)",
            ]
        )
        plt.ylim(-0.05, 1.05)
        plt.xlim(-0.05 * len(y_proba), 1.05 * len(y_proba))
        plt.savefig(os.path.join(root_dir, f"{file_name}.png"), bbox_inches='tight', dpi=600)
        plt.savefig(os.path.join(root_dir, f"{file_name}.pdf"), bbox_inches='tight')

        ############################# ------------------------------------------------

        import matplotlib.pyplot as plt
        import seaborn as sns

        # y_pred = np.where(s == 1, 1, np.where(y_proba > (1 + s_proba) / 2, 1, 0))
        # y_alt = np.where(s == 1, 1, np.where(y_proba > 0.5, 1, 0))
        y_pred = np.where(s == 1, 1, np.where(corrected_rule > 0.5, 1, 0))
        y_alt = np.where(s == 1, 1, np.where(y_proba > 0.5, 1, 0))

        sns.set_theme()
        plt.figure(figsize=(8, 5))
        idx = np.argsort(y_proba)
        plt.plot(np.arange(-len(idx), 2 * len(idx)), np.repeat(0.5, 3 * len(idx)), "k.", ms=1)
        plt.plot(np.arange(len(idx)), y_proba[idx], "b.", ms=10)

        same_clf = (y_pred == y_alt)[idx]
        diff_clf = (y_pred != y_alt)[idx]

        plt.plot(
            np.arange(len(idx))[same_clf],
            (corrected_rule)[idx][same_clf],
            "ro",
            ms=2,
            alpha=0.7,
        )
        # plt.plot(
        #     np.arange(len(idx))[same_clf][y[idx][same_clf] == 0],
        #     (corrected_rule)[idx][same_clf][y[idx][same_clf] == 0],
        #     "rx",
        #     ms=2,
        #     alpha=0.3,
        # )

        plt.plot(
            np.arange(len(idx))[diff_clf],
            (corrected_rule)[idx][diff_clf],
            "go",
            ms=3,
            alpha=1,
        )
        # plt.plot(
        #     np.arange(len(idx))[diff_clf][y[idx][diff_clf] == 0],
        #     (corrected_rule)[idx][diff_clf][y[idx][diff_clf] == 0],
        #     "gx",
        #     ms=3,
        #     alpha=0.5,
        # )

        x_classification_change = np.where(y_proba[idx] > 0.5)[0]
        plt.fill_between(np.concatenate([x_classification_change, np.arange(np.max(x_classification_change), 2 * np.max(x_classification_change))]), -1, 0.5, color='#FFC310', alpha=0.5)

        plt.legend(
            [
                "_nolegend_",
                "$y(x)$",
                "$y(x) - \\frac{s(x)}{2}$ (same class)",
                "$y(x) - \\frac{s(x)}{2}$ (classification changed)",
            ]
        )
        plt.ylim(-0.05, 1.05)
        plt.xlim(-0.05 * len(y_proba), 1.05 * len(y_proba))
        plt.savefig(os.path.join(root_dir, f"{file_name} clf.png"), bbox_inches='tight', dpi=600)
        plt.savefig(os.path.join(root_dir, f"{file_name} clf.pdf"), bbox_inches='tight')

# %%





































# %%
# calculate_metrics
#     import matplotlib.pyplot as plt
#     import pandas as pd
#     import seaborn as sns

#     y_pred = np.where(s == 1, 1, np.where(y_proba > (1 + s_proba) / 2, 1, 0))
#     y_alt = np.where(s == 1, 1, np.where(y_proba > 0.5, 1, 0))

#     sns.set_theme()
#     plt.figure(figsize=(8, 5))
#     idx = np.argsort(y_proba)
#     plt.plot(np.arange(-len(idx), 2 * len(idx)), np.repeat(0.5, 3 * len(idx)), "k.", ms=1)
#     plt.plot(np.arange(len(idx)), y_proba[idx], "b.", ms=10)


#     plt.plot(
#         np.arange(len(idx))[(s[idx] == 1) & (y[idx] == 1)],
#         (y_proba - s_proba / 2)[idx][(s[idx] == 1) & (y[idx] == 1)],
#         "r.",
#         ms=3,
#         alpha=0.5,
#     )

#     plt.plot(
#         np.arange(len(idx))[(s[idx] == 0) & (y[idx] == 1)],
#         (y_proba - s_proba / 2)[idx][(s[idx] == 0) & (y[idx] == 1)],
#         "g.",
#         ms=3,
#         alpha=0.5,
#     )
#     plt.plot(
#         np.arange(len(idx))[(s[idx] == 0) & (y[idx] == 0)],
#         (y_proba - s_proba / 2)[idx][(s[idx] == 0) & (y[idx] == 0)],
#         "gx",
#         ms=3,
#         alpha=0.5,
#     )

#     x_classification_change = np.where(y_proba[idx] > 0.5)[0]
#     plt.fill_between(np.concatenate([x_classification_change, np.arange(np.max(x_classification_change), 2 * np.max(x_classification_change))]), -1, 0.5, color='#FFC310', alpha=0.5)

#     plt.legend(
#         [
#             "_nolegend_",
#             "$y(x)$",
#             "$y(x) - \\frac{s(x)}{2}$ ($S = 1$)",
#             "$y(x) - \\frac{s(x)}{2}$ ($S = 0, Y = 1$)",
#             "$y(x) - \\frac{s(x)}{2}$ ($S = 0, Y = -1$)",
#         ]
#     )
#     plt.ylim(-0.05, 1.05)
#     plt.xlim(-0.05 * len(y_proba), 1.05 * len(y_proba))
#     plt.savefig("diagnostic-plots/s-split.png", bbox_inches='tight', dpi=600)
#     plt.savefig("diagnostic-plots/s-split.pdf", bbox_inches='tight')

# #############################

#     import matplotlib.pyplot as plt
#     import seaborn as sns

#     y_pred = np.where(s == 1, 1, np.where(y_proba > (1 + s_proba) / 2, 1, 0))
#     y_alt = np.where(s == 1, 1, np.where(y_proba > 0.5, 1, 0))

#     sns.set_theme()
#     plt.figure(figsize=(8, 5))
#     idx = np.argsort(y_proba)
#     plt.plot(np.arange(-len(idx), 2 * len(idx)), np.repeat(0.5, 3 * len(idx)), "k.", ms=1)
#     plt.plot(np.arange(len(idx)), y_proba[idx], "b.", ms=10)

#     same_clf = (y_pred == y_alt)[idx]
#     diff_clf = (y_pred != y_alt)[idx]

#     plt.plot(
#         np.arange(len(idx))[same_clf][y[idx][same_clf] == 1],
#         (y_proba - s_proba / 2)[idx][same_clf][y[idx][same_clf] == 1],
#         "ro",
#         ms=2,
#         alpha=0.3,
#     )
#     plt.plot(
#         np.arange(len(idx))[same_clf][y[idx][same_clf] == 0],
#         (y_proba - s_proba / 2)[idx][same_clf][y[idx][same_clf] == 0],
#         "rx",
#         ms=2,
#         alpha=0.3,
#     )

#     plt.plot(
#         np.arange(len(idx))[diff_clf][y[idx][diff_clf] == 1],
#         (y_proba - s_proba / 2)[idx][diff_clf][y[idx][diff_clf] == 1],
#         "go",
#         ms=3,
#         alpha=0.5,
#     )
#     plt.plot(
#         np.arange(len(idx))[diff_clf][y[idx][diff_clf] == 0],
#         (y_proba - s_proba / 2)[idx][diff_clf][y[idx][diff_clf] == 0],
#         "gx",
#         ms=3,
#         alpha=0.5,
#     )

#     x_classification_change = np.where(y_proba[idx] > 0.5)[0]
#     plt.fill_between(np.concatenate([x_classification_change, np.arange(np.max(x_classification_change), 2 * np.max(x_classification_change))]), -1, 0.5, color='#FFC310', alpha=0.5)

#     plt.legend(
#         [
#             "_nolegend_",
#             "$y(x)$",
#             "$y(x) - \\frac{s(x)}{2}$ (same class, Y = 1)",
#             "$y(x) - \\frac{s(x)}{2}$ (same class, Y = -1)",
#             "$y(x) - \\frac{s(x)}{2}$ (classification changed, Y = 1)",
#             "$y(x) - \\frac{s(x)}{2}$ (classification changed, Y = -1)",
#         ]
#     )
#     plt.ylim(-0.05, 1.05)
#     plt.xlim(-0.05 * len(y_proba), 1.05 * len(y_proba))
#     plt.savefig("diagnostic-plots/classification-change-split.png", bbox_inches='tight', dpi=600)
#     plt.savefig("diagnostic-plots/classification-change-split.pdf", bbox_inches='tight')
