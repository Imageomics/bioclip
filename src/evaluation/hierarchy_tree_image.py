import os
from datetime import timedelta
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision.datasets.folder import find_classes

from evaluation.utils import load_json, save_json

TAXONOMICAL_RANKS = [
    "Kingdoms",
    "Phyla",
    "Classes",
    "Orders",
    "Families",
    "Genera",
    "Species",
]

COLORS = [
    (127, 60, 141),
    (17, 165, 121),
    (57, 105, 172),
    (242, 183, 1),
    (231, 63, 116),
    (128, 186, 90),
    (230, 131, 16),
    (0, 134, 149),
    (207, 28, 144),
    (249, 123, 114),
    (165, 170, 153),
]


def get_colors():
    colors = []
    for c in COLORS:
        colors.append((c[0] / 255, c[1] / 255, c[2] / 255))
    return colors


def mahalanobis(x):
    mu = x.mean(0)
    cov = np.cov(x.T)

    x_mu = x - mu
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(x_mu, inv_covmat)
    mahal = np.dot(left, x_mu.T)
    return mahal.diagonal()


def create_hierarchical_tree_vis(
    features,
    hierarchy_labels,
    img_paths,
    reduction_method="tsne",
    hierarchy_label_map=None,
    top_k=6,
    output="",
    rerun_reduction=False,
    verbose=False,
    remove_outliers=False,
    precomputed_reductions=None,
    dpi=100,
):
    assert reduction_method in ["tsne", "pca"], f"{reduction_method} not supported."
    os.makedirs(output, exist_ok=True)

    def log(x):
        if verbose:
            print(x)

    lvls = len(hierarchy_labels[0])
    base_colors = get_colors()

    def run_reduction(in_feats):
        log(
            f"Running {reduction_method} on features: {in_feats.shape}. This may take some time."
        )
        start = timer()
        out_feats = reduce.fit_transform(in_feats)
        end = timer()
        log(f"{reduction_method} completed: {timedelta(seconds=end-start)}")

        return out_feats

    if reduction_method == "tsne":
        reduce = TSNE(n_jobs=16, n_components=2, verbose=2)
    elif reduction_method == "pca":
        reduce = PCA(2)

    if not rerun_reduction:
        if precomputed_reductions is None:
            features = run_reduction(features)
        else:
            features = precomputed_reductions
        saved_features = features
        saved_labels = hierarchy_labels
        saved_paths = img_paths
    else:
        saved_features = []
        saved_labels = []
        saved_paths = []

    data_queue = [(features, hierarchy_labels, img_paths, 0)]
    largest_name = None
    graph_data = []

    nrows = 3
    ncols = 3
    height = 7
    width = 5
    final_fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(height * nrows, width * ncols), dpi=dpi
    )
    while len(data_queue) > 0:
        past_largest_name = largest_name
        cur_plot_lvl = past_largest_name if past_largest_name is not None else "Kingdom"

        feats, lbls, paths, lvl = data_queue.pop(0)
        if rerun_reduction:
            if precomputed_reductions is None:
                reduced_feats = run_reduction(feats)
            else:
                reduced_feats = precomputed_reductions[lvl]
            saved_features.append(reduced_feats)
            saved_labels.append(lbls)
            saved_paths.append(paths)
        else:
            reduced_feats = feats
        if hierarchy_label_map is not None:
            lvl_lbl_map = hierarchy_label_map[lvl]

        log(f"Plotting Level {cur_plot_lvl}")

        lbl_lengths = []
        sorted_lbls = sorted(list(set(lbls[:, lvl])))
        for lbl in sorted_lbls:
            idx = lbls[:, lvl] == lbl
            lbl_lengths.append([lbl, len(reduced_feats[idx])])
        lbl_lengths = sorted(lbl_lengths, key=lambda x: x[1], reverse=True)

        row = lvl // ncols
        col = lvl % ncols
        if lvl == 6:
            col += 1
        ax = axs[row, col]
        plt.setp(ax.spines.values(), lw=3, color="black")  # Set border width
        title_parts = cur_plot_lvl.split("_")
        fig_title = TAXONOMICAL_RANKS[lvl]
        if lvl > 0:
            fig_title += f" of {title_parts[-1]}"
        ax.set_title(f"({lvl+1}) {fig_title}", fontsize=25, y=1.02)

        most_feats = None
        most_lbls = None
        highest_num = 0
        c = 0
        for lbl in sorted_lbls:
            if top_k > 0:
                if lbl not in np.array(lbl_lengths)[:top_k, 0]:
                    continue
            idx = lbls[:, lvl] == lbl
            feat = reduced_feats[idx]
            if hierarchy_label_map is not None:
                name = lvl_lbl_map[str(lbl)]
            else:
                name = f"{lbl.split('_')[-1]}"

            if len(feat) > highest_num:
                highest_num = len(feat)
                if rerun_reduction:
                    most_feats = feats[idx]
                else:
                    most_feats = feat
                most_lbls = lbls[idx]
                most_paths = np.array(paths)[idx].tolist()
                largest_name = lbl
                if hierarchy_label_map is not None:
                    largest_name = lvl_lbl_map[lbl]

            plot_feat = feat
            if remove_outliers:
                mah_dist = mahalanobis(feat)
                # https://www.statology.org/mahalanobis-distance-python/
                # Typically a p-value that is < .001 is considered an outlier
                p = 1 - chi2.cdf(mah_dist, feat.shape[1] - 1)
                filtered_idx = p >= 0.001
                plot_feat = feat[filtered_idx]

            markersize = 2
            if lvl == 5:
                markersize *= 2
            elif lvl == 6:
                markersize *= 4
            ax.scatter(
                plot_feat[:, 0],
                plot_feat[:, 1],
                label=name,
                color=base_colors[c],
                alpha=0.50,
                s=markersize,
                rasterized=True,
            )  # have to rasterize for .pdf to load quicker

            c += 1

        # Add to queue
        if (lvl + 1) < lvls:
            data_queue.append((most_feats, most_lbls, most_paths, lvl + 1))

        if lvl > 0:
            save_path = os.path.join(output, f"depth_{lvl}_{past_largest_name}.png")
        else:
            save_path = os.path.join(output, f"depth_{lvl}.png")

        markerscale = 8
        if lvl == 5:
            markerscale = 6
        elif lvl == 6:
            markerscale = 4
        ax.legend(loc="upper right", ncols=2, markerscale=markerscale, fontsize=12)
        # Turn off axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

        ax_extent = ax.get_window_extent().transformed(
            final_fig.dpi_scale_trans.inverted()
        )
        final_fig.savefig(save_path, bbox_inches=ax_extent)
        graph_data.append((save_path, lvl))

    for r in range(nrows):
        for c in range(ncols):
            axs[r][c].set_xticks([])
            axs[r][c].set_yticks([])
    plt.setp(axs[-1][-1].spines.values(), lw=0)
    plt.setp(axs[-1][-3].spines.values(), lw=0)

    final_fig.tight_layout(h_pad=1, w_pad=1)
    final_fig.savefig(os.path.join(output, "full_image.png"))
    final_fig.savefig(os.path.join(output, "full_image.pdf"))

    return saved_features, saved_labels, saved_paths


def _get_hierarchy_lbl_map(root):
    classes, class_to_idx = find_classes(root)
    idx_to_class = {val: key for (key, val) in class_to_idx.items()}

    return idx_to_class


def _get_args():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--val_root", type=str, default="/local/scratch/cv_datasets/inat21/raw/val"
    )
    parser.add_argument(
        "--features_output",
        type=str,
        default="/local/scratch/carlyn.1/clip_paper_bio/features",
    )
    parser.add_argument("--results_output", type=str, default="tmp")
    parser.add_argument("--exp_type", type=str, default="8_25_2023_83_epochs")
    parser.add_argument(
        "--reduction", type=str, default="tsne", choices=["tsne", "pca"]
    )
    parser.add_argument("--rerun", action="store_true", default=False)
    parser.add_argument("--remove_outliers", action="store_true", default=False)
    parser.add_argument("--subset", type=int, default=-1)
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--dpi", type=float, default=500)

    return parser.parse_args()


if __name__ == "__main__":
    args = _get_args()

    hierarchy_label_map = _get_hierarchy_lbl_map(args.val_root)

    features = np.load(
        os.path.join(args.features_output, args.exp_type + "_features.npy")
    )
    labels = np.load(os.path.join(args.features_output, args.exp_type + "_labels.npy"))
    img_paths = load_json(
        os.path.join(args.features_output, args.exp_type + "_paths.json")
    )

    hierarchy_labels = []
    for lbl in labels:
        lvls = hierarchy_label_map[lbl].split("_")[1:]
        hierarchy_labels.append(
            ["_".join(lvls[: lvl + 1]) for lvl in range((len(lvls)))]
        )

    hierarchy_labels = np.array(hierarchy_labels)

    folder_name = (
        f"{args.exp_type}_{args.reduction}_{'rerun' if args.rerun else 'no_rerun'}"
    )
    folder_name += f"_top_{args.top_k}_{'remove_outliers' if args.remove_outliers else 'outliers_remain'}"

    output_name = os.path.join(args.results_output, folder_name)

    if args.subset > 0:
        features = features[: args.subset]
        hierarchy_labels = hierarchy_labels[: args.subset]
        img_paths = img_paths[: args.subset]

    precomputed_features = None
    precompute_name = "precomputed_0.npy" if args.rerun else "precomputed.npy"
    precomputed_lbl_name = (
        "precomputed_0_labels.npy" if args.rerun else "precomputed_labels.npy"
    )
    precomputed_path_name = (
        "precomputed_0_paths.json" if args.rerun else "precomputed_paths.json"
    )
    precomputed_path = os.path.join(output_name, precompute_name)
    precomputed_label_path = os.path.join(output_name, precomputed_lbl_name)
    precomputed_path_path = os.path.join(output_name, precomputed_path_name)
    precomputed_path_path = os.path.join(output_name, precomputed_path_name)
    if os.path.exists(precomputed_path):
        if not args.rerun:
            precomputed_features = np.load(precomputed_path)
        else:
            precomputed_features = []
            for i in range(7):
                precomputed_features.append(
                    np.load(os.path.join(output_name, f"precomputed_{i}.npy"))
                )

    sf, sl, sp = create_hierarchical_tree_vis(
        features,
        hierarchy_labels,
        img_paths,
        reduction_method=args.reduction,
        output=output_name,
        rerun_reduction=args.rerun,
        verbose=True,
        top_k=args.top_k,
        remove_outliers=args.remove_outliers,
        precomputed_reductions=precomputed_features,
        dpi=args.dpi,
    )

    if not args.rerun:
        np.save(precomputed_path, sf)
        np.save(precomputed_label_path, sl)
        np.save(precomputed_path_path, sp)
    else:
        for i, f in enumerate(sf):
            np.save(os.path.join(output_name, f"precomputed_{i}.npy"), f)
        for i, l in enumerate(sl):
            np.save(os.path.join(output_name, f"precomputed_{i}_labels.npy"), l)
        for i, p in enumerate(sp):
            save_json(os.path.join(output_name, f"precomputed_{i}_paths.json"), p)
