import os
from argparse import ArgumentParser

import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, pairwise_distances
from torchvision import transforms as T

from evaluation.hierarchy_tree_image import get_colors
from evaluation.utils import load_json


def _get_arrow_end(img_ax, arrow_end_code):
    xmin = img_ax.bbox.xmin
    xmax = img_ax.bbox.xmax
    ymin = img_ax.bbox.ymin
    ymax = img_ax.bbox.ymax

    if arrow_end_code == "cr":
        return xmax, (ymin + ymax) / 2
    elif arrow_end_code == "br":
        return xmax, ymin
    elif arrow_end_code == "bc":
        return (xmin + xmax) / 2, ymin
    elif arrow_end_code == "bl":
        return xmin, ymin
    elif arrow_end_code == "cl":
        return xmin, (ymin + ymax) / 2
    else:
        assert False, f"{arrow_end_code} is not a valid arrow end code"


def _load_img(path):
    resize_fn = T.Resize((256, 256), Image.BILINEAR)
    im = Image.open(path)
    im = resize_fn(im)
    im = np.array(im).astype(np.float32) / 255

    return im


def _get_rep_imgs(point, tsne_feautres, img_paths, num_imgs=4):
    x, y = point
    in_point = np.array(point)[np.newaxis, :]
    dist = pairwise_distances(tsne_feautres, in_point)[:, 0]
    idx = np.argsort(dist)

    return np.array(img_paths)[idx[:num_imgs]].tolist()


def create_zoom_figure(
    tsne_feautres,
    high_level_labels,
    img_paths,
    zoom_points,
    rep_spec_idx,
    arrow_end_codes,
    output_path,
    output_name,
    lvl,
    dpi=500,
    top_k=6,
    num_imgs=9,
):
    """
    tsne_feautres: 2-dim coordinates from TSNE
    high_level_labels: labels for the next tier. If we have Orders of insecta, then we should have the labels be those orders
    img_paths: The associated image paths
    zoom_points: points in the plot to show zoomed images
    rep_spec_idx: 1-to-1 with zoom_points. Where to place zoom rep images
    arrow_end_codes: where the end of the arrow should go
    output_path: where to save the figure
    output_name: what to name the figure
    lvl: the hierarchy level being visualized
    dpi: dots-per-inch of image
    top_k: top categories to visualize
    num_imgs: # of images to zoom into
    """

    base_colors = get_colors()

    fig_width = 15
    fig_height = 7
    final_fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    final_fig.set_tight_layout(True)
    final_spec = gridspec.GridSpec(3, 6)

    lbl_lengths = []
    sorted_lbls = sorted(list(set(high_level_labels[:, lvl])))
    for lbl in sorted_lbls:
        idx = high_level_labels[:, lvl] == lbl
        lbl_lengths.append([lbl, len(tsne_feautres[idx])])
    lbl_lengths = sorted(lbl_lengths, key=lambda x: x[1], reverse=True)

    plot_ax = final_fig.add_subplot(final_spec[1:, 1:5])

    c = 0
    for lbl in sorted_lbls:
        if top_k > 0:
            if lbl not in np.array(lbl_lengths)[:top_k, 0]:
                continue
        idx = high_level_labels[:, lvl] == lbl
        feat = tsne_feautres[idx]
        name = f"{lbl.split('_')[-1]}"

        markersize = 2

        # have to rasterize for .pdf to load quicker
        plot_ax.scatter(
            feat[:, 0],
            feat[:, 1],
            label=name,
            color=base_colors[c],
            alpha=0.50,
            s=markersize,
            rasterized=True,
        )
        c += 1

    plot_ax.legend(
        bbox_to_anchor=(0.5, 0.0),
        loc="upper center",
        ncols=6,
        markerscale=8,
        fontsize=12,
    )

    # TSNE plot axis
    plot_ax.set_xticks([])
    plot_ax.set_yticks([])
    plt.setp(plot_ax.spines.values(), lw=0)

    zoom_plot_ax = []
    for i, (x, y) in enumerate(zoom_points):
        # Add Ellipse
        fig_ax = plt.gca()
        ew = 4
        eh = ew * (fig_width / fig_height)
        elip = patches.Ellipse(
            [x, y],
            height=eh,
            width=ew,
            linewidth=2,
            fill=False,
            color="black",
            transform=plot_ax.transData,
        )
        plot_ax.add_patch(elip)

        # Get Representative images
        rep_img_paths = _get_rep_imgs(
            [x, y], tsne_feautres, img_paths, num_imgs=num_imgs
        )
        ims = [_load_img(p) for p in rep_img_paths]
        rep_img = np.concatenate(
            (
                np.concatenate(ims[:3], axis=1),
                np.concatenate(ims[3:6], axis=1),
                np.concatenate(ims[6:], axis=1),
            ),
            axis=0,
        )

        # Plot Representative images
        sx, sy = rep_spec_idx[i]
        img_ax = final_fig.add_subplot(final_spec[sx, sy])
        img_ax.imshow(rep_img)
        img_ax.set_xticks([])
        img_ax.set_yticks([])
        plt.setp(img_ax.spines.values(), lw=0)

        zoom_plot_ax.append(img_ax)

    final_fig.tight_layout()
    final_spec.tight_layout(figure=final_fig)

    # Plot arrows after final image
    for i, (x, y) in enumerate(zoom_points):
        ax = zoom_plot_ax[i]
        rep_x, rep_y = _get_arrow_end(ax, arrow_end_codes[i])
        fig_ax.annotate(
            "",
            xy=(rep_x, rep_y),
            xytext=(x, y),
            xycoords=transforms.IdentityTransform(),
            textcoords=plot_ax.transData,
            arrowprops=dict(arrowstyle="]->", shrinkA=20, shrinkB=10, linewidth=2),
        )

    final_fig.savefig(os.path.join(output_path, f"{output_name}.png"))
    final_fig.savefig(os.path.join(output_path, f"{output_name}.pdf"))

    """
    plt.setp(bio_ax.spines.values(), lw=3, color='black') # Set border width
    plt.setp(oa_ax.spines.values(), lw=3, color='black') # Set border width

    """


def _get_zoom_fig_seeds(exp_type, lvl):
    # TODO: levels 5 and 6 need to seeded properly
    if exp_type == 'bioclip':
        zoom_data = {
            0 : {
                "zoom_points"     : [(-56, -71), (-90, -17), (-80, 57), (-23, -40), (-25, 47), (12, 95), (23, 30), (63, 49), (27, -25), (50, -80)],
                "rep_spec_idx"    : [(2, 0),     (1, 0),     (0, 0),    (0, 1),     (0, 2),   (0, 3),   (0, 4),   (0, 5),   (1, 5),    (2, 5)],
                "arrow_end_codes" : ['cr',       'cr',       'br',      'bc',       'bc',     'bc',     'bl',     'bl',     'cl',      'cl']
            },
            1 : {
                "zoom_points"     : [(-57, -67), (-90, -20), (-83, 60), (-27, -36), (-20, 0), (11, 100), (23, 30), (75, 37), (82, 30), (65, 15)],
                "rep_spec_idx"    : [(2, 0),     (1, 0),     (0, 0),    (0, 1),     (0, 2),   (0, 3),   (0, 4),   (0, 5),   (1, 5),    (2, 5)],
                "arrow_end_codes" : ['cr',       'cr',       'br',      'bc',       'bc',     'bc',     'bl',     'bl',     'cl',      'cl']
            },
            2 : {
                "zoom_points"     : [(-32, -86), (-87, -44), (-56, -27), (-57, -19), (-63, -30), (5, 95), (31, 20), (66, 54), (30, -20), (60, -80)],
                "rep_spec_idx"    : [(2, 0),     (1, 0),     (0, 1),    (0, 2),     (0, 0),   (0, 3),   (0, 4),   (0, 5),   (1, 5),    (2, 5)],
                "arrow_end_codes" : ['cr',       'cr',       'bc',      'bc',       'br',     'bc',     'bl',     'bl',     'cl',      'cl']
            },
            3 : {
                "zoom_points"     : [(-56, -71), (-90, -40), (-38, 48), (-23, -40), (-20, 0), (12, 95), (23, 10), (63, 49), (27, -20), (50, -80)],
                "rep_spec_idx"    : [(2, 0),     (1, 0),     (0, 1),    (0, 0),     (0, 2),   (0, 3),   (0, 4),   (0, 5),   (1, 5),    (2, 5)],
                "arrow_end_codes" : ['cr',       'br',       'bc',      'br',       'bc',     'bc',     'bl',     'bl',     'cl',      'cl']
            },
            4 : {
                "zoom_points"     : [(-17, -67), (-70, -30), (-54, 31), (-30, -18), (-18, 53), (15, 89), (29, 10), (62, 41), (28, -19), (47, -80)],
                "rep_spec_idx"    : [(2, 0),     (1, 0),     (0, 0),    (0, 1),     (0, 2),   (0, 3),   (0, 4),   (0, 5),   (1, 5),    (2, 5)],
                "arrow_end_codes" : ['cr',       'cr',       'br',      'br',       'bc',     'bc',     'bl',     'bl',     'cl',      'cl']
            },
            5 : {
                "zoom_points"     : [(-56, -71), (-90, -40), (-38, 48), (-23, -40), (-20, 0), (12, 95), (23, 10), (63, 49), (27, -20), (50, -80)],
                "rep_spec_idx"    : [(2, 0),     (1, 0),     (0, 0),    (0, 1),     (0, 2),   (0, 3),   (0, 4),   (0, 5),   (1, 5),    (2, 5)],
                "arrow_end_codes" : ['cr',       'br',       'bc',      'br',       'bc',     'bc',     'bl',     'bl',     'cl',      'cl']
            },
            6 : {
                "zoom_points"     : [(-56, -71), (-90, -40), (-38, 48), (-23, -40), (-20, 0), (12, 95), (23, 10), (63, 49), (27, -20), (50, -80)],
                "rep_spec_idx"    : [(2, 0),     (1, 0),     (0, 0),    (0, 1),     (0, 2),   (0, 3),   (0, 4),   (0, 5),   (1, 5),    (2, 5)],
                "arrow_end_codes" : ['cr',       'br',       'bc',      'br',       'bc',     'bc',     'bl',     'bl',     'cl',      'cl']
            }
        }

    else:
        zoom_data = {
            0 : {
                "zoom_points"     : [(-28, -25), (-84, 12), (-70, 37), (-37, 31), (-7, 70), (10, 79), (25, 0), (60, 60), (64, -5), (50, -70)],
                "rep_spec_idx"    : [(2, 0),     (1, 0),    (0, 0),    (0, 1),    (0, 2),    (0, 3),   (0, 4),   (0, 5),   (1, 5),    (2, 5)],
                "arrow_end_codes" : ['cr',       'cr',      'br',      'bc',      'bc',      'bc',     'bl',     'bl',     'cl',      'cl']
            },
            1 : {
                "zoom_points"     : [(-53, -32), (-84, 12), (-70, 37), (-33, 31), (9, 87), (22, 82), (28, 72), (52, 58), (64, -5), (50, -70)],
                "rep_spec_idx"    : [(2, 0),     (1, 0),    (0, 0),    (0, 1),    (0, 2),    (0, 3),   (0, 4),   (0, 5),   (1, 5),    (2, 5)],
                "arrow_end_codes" : ['cr',       'cr',      'br',      'bc',      'bc',      'bc',     'bl',     'bl',     'cl',      'cl']
            },
            2 : {
                "zoom_points"     : [(-52, -32), (-84, 12), (-65, 37), (-30, 71), (-17, 90), (-6, 86), (9, 48), (56, 30), (11, 12), (20, -70)],
                "rep_spec_idx"    : [(2, 0),     (1, 0),    (0, 0),    (0, 1),    (0, 2),    (0, 3),   (0, 4),   (1, 5),   (0, 5),    (2, 5)],
                "arrow_end_codes" : ['cr',       'cr',      'br',      'bc',      'bc',      'bc',     'bl',     'cl',     'bl',      'cl']
            },
            3 : {
                "zoom_points"     : [(-58, -25), (-64, 12), (-41, 37), (-25, 34), (-19, 19), (10, 65), (25, 15), (60, 34), (55, -10), (20, -70)],
                "rep_spec_idx"    : [(2, 0),     (1, 0),    (0, 0),    (0, 1),    (0, 2),    (0, 3),   (0, 4),   (0, 5),   (1, 5),    (2, 5)],
                "arrow_end_codes" : ['cr',       'cr',      'br',      'bc',      'bc',      'bc',     'bl',     'bl',     'cl',      'cl']
            },
            4 : {
                "zoom_points"     : [(-44, -37), (-59, 3), (-52, 27), (-38, 38), (-28, 13), (-5, 72), (8, 19), (57, 32), (54, -11), (19, -52)],
                "rep_spec_idx"    : [(2, 0),     (1, 0),    (0, 0),    (0, 1),    (0, 2),    (0, 3),   (0, 4),   (0, 5),   (1, 5),    (2, 5)],
                "arrow_end_codes" : ['cr',       'cr',      'br',      'bc',      'bc',      'bc',     'bl',     'bl',     'cl',      'cl']
            },
            5 : {
                "zoom_points"     : [(-44, -37), (-59, 3), (-52, 27), (-38, 38), (-28, 13), (-5, 72), (8, 19), (57, 32), (54, -11), (19, -52)],
                "rep_spec_idx"    : [(2, 0),     (1, 0),    (0, 0),    (0, 1),    (0, 2),    (0, 3),   (0, 4),   (0, 5),   (1, 5),    (2, 5)],
                "arrow_end_codes" : ['cr',       'cr',      'br',      'bc',      'bc',      'bc',     'bl',     'bl',     'cl',      'cl']
            },
            6 : {
                "zoom_points"     : [(-44, -37), (-59, 3), (-52, 27), (-38, 38), (-28, 13), (-5, 72), (8, 19), (57, 32), (54, -11), (19, -52)],
                "rep_spec_idx"    : [(2, 0),     (1, 0),    (0, 0),    (0, 1),    (0, 2),    (0, 3),   (0, 4),   (0, 5),   (1, 5),    (2, 5)],
                "arrow_end_codes" : ['cr',       'cr',      'br',      'bc',      'bc',      'bc',     'bl',     'bl',     'cl',      'cl']
            }
        }

    return zoom_data[lvl]["zoom_points"], zoom_data[lvl]["rep_spec_idx"], zoom_data[lvl]["arrow_end_codes"]


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--exp_type", type=str, default="bioclip", choices=["bioclip", "openai"]
    )
    parser.add_argument("--lvl", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--output_dir", type=str, default="tmp/")

    return parser.parse_args()


def _get_lvl_cls(path, lvl, is_img_path=False):
    if lvl == 0:
        return "Root"
    if is_img_path:
        lvls = path.split(os.path.sep)[-2]
    else:
        lvls = path
    return lvls.split("_")[lvl]


def _calc_acc(pred_list_path, lvl, lvl_cls):
    filtered = []
    correct = 0
    total = 0
    with open(pred_list_path) as f:
        lines = f.readlines()
        for line in lines[1:]:
            path, pred_y, tgt_y, pred_cls, tgt_cls = line.split(",")
            if pred_y == tgt_y:
                correct += 1
            total += 1
            tgt_lvl_cls = _get_lvl_cls(tgt_cls, lvl)
            if tgt_lvl_cls == lvl_cls:
                filtered.append([path, pred_y, tgt_y, pred_cls, tgt_cls])

    print(f"Zero shot accuracy on iNat21: {round(correct / total, 4) * 100}%")
    filtered = np.array(filtered)
    matrix = confusion_matrix(filtered[:, 4], filtered[:, 3])
    print(matrix)
    identity = np.identity(n=matrix.shape[0])
    max_idx_x, max_idx_y = np.unravel_index(np.argmax(matrix * identity), matrix.shape)
    print(matrix[max_idx_x, max_idx_y])
    exit()


if __name__ == "__main__":
    args = get_args()

    output_path = args.output_dir
    output_name = f"intrinsic_zoom_{args.exp_type}_lvl_{args.lvl}"

    if args.exp_type == "bioclip":
        base_path = "tmp/11_2023_tsne_rerun_top_6_remove_outliers"
        pred_list_path = (
            "logs/tol1m-random-best-val-few-shot/inat21-predictions-bioclip.csv"
        )

    else:
        base_path = "tmp/openai_pretrain_tsne_rerun_top_6_remove_outliers"
        pred_list_path = (
            "logs/tol1m-random-best-val-few-shot/inat21-predictions-openai.csv"
        )

    zoom_points, rep_spec_idx, arrow_end_codes = _get_zoom_fig_seeds(
        args.exp_type, args.lvl
    )

    tsne_feautres = np.load(os.path.join(base_path, f"precomputed_{args.lvl}.npy"))
    high_level_labels = np.load(
        os.path.join(base_path, f"precomputed_{args.lvl}_labels.npy")
    )
    img_paths = load_json(os.path.join(base_path, f"precomputed_{args.lvl}_paths.json"))

    accuracies = _calc_acc(
        pred_list_path, args.lvl, _get_lvl_cls(img_paths[0], args.lvl, is_img_path=True)
    )

    create_zoom_figure(
        tsne_feautres,
        high_level_labels,
        img_paths,
        zoom_points,
        rep_spec_idx,
        arrow_end_codes,
        output_path,
        output_name,
        args.lvl,
        dpi=args.dpi,
    )
