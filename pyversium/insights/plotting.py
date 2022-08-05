import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from sklearn import metrics
from sklearn.calibration import CalibrationDisplay, calibration_curve
from sklearn.experimental import enable_halving_search_cv  # noqa

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")


def _make_lift_dict(y_true, y_score):
    if len(y_score.shape) > 1 and y_score.shape[1] > 1:
        y_score = y_score[:, 1]

    # Calculate data for plots
    score_df = pd.DataFrame(data={"scores": y_score, "y_true": y_true})
    score_df["decile"] = pd.qcut(score_df["scores"], 10,
                                  labels=["1-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90", "91-100"])

    gains = score_df.groupby("decile")['y_true'].sum() / (score_df['y_true'].sum() / 10)
    return gains.reset_index().to_dict(orient='list')


def _make_roc_dict(y_true, y_score, pos_label=None):
    if len(y_score.shape) > 1 and y_score.shape[1] > 1:
        y_score = y_score[:, 1]


    roc_dict = {'pos_label': pos_label}
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=roc_dict['pos_label'])
    roc_dict['fpr'] = fpr
    roc_dict['tpr'] = tpr
    roc_dict['threshold'] = thresholds
    roc_dict['roc_auc'] = metrics.roc_auc_score(y_true, y_score)
    return roc_dict

def _make_calibration_dict(y_true, y_score, pos_label=None):
    if len(y_score.shape) > 1 and y_score.shape[1] > 1:
        y_score = y_score[:, 1]

    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10,
                                             pos_label=pos_label)

    cal_dict = {'pos_label': pos_label,
                'prob_true': prob_true,
                'prob_pred': prob_pred,
                'y_prob': y_score}
    return cal_dict


def _plot_lift_chart(gains_dict, figure=None):
    figure = figure if figure is not None else plt.figure(figsize=(10, 7), dpi=200, tight_layout=True)
    gains = pd.DataFrame(gains_dict)
    gains['dotted_line'] = 1.0
    gains['color'] = 'r'
    gains.loc[gains.y_true > 1.0, 'color'] = 'g'

    ax = plt.axes()
    barplot = ax.barh(gains['decile'], gains['y_true'], color=gains.color, height=1)

    new_patches = []
    for patch in reversed(ax.patches):
        bb = patch.get_bbox()
        color = patch.get_facecolor()
        p_bbox = FancyBboxPatch((bb.xmin, bb.ymin),
                                abs(bb.width), abs(bb.height),
                                boxstyle="round,pad=-0.0040,rounding_size=0.015",
                                ec="none", fc=color,
                                mutation_aspect=4
                                )
        patch.remove()
        new_patches.append(p_bbox)
    for patch in new_patches:
        ax.add_patch(patch)
    y_limits = np.linspace(*ax.get_ylim())
    average_line = ax.plot([1.0] * len(y_limits), y_limits, 'k--', label="Average Lift")
    ax.set_title("Lift Chart")
    ax.set_ylabel("Score Range (Deciles)")
    ax.set_xlabel("Lift")

    colors = {'Worse Than Average': 'red', 'Better than Average': 'green'}
    legend_labels = list(colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in legend_labels] + average_line
    legend_labels += ["Average Lift"]
    ax.legend(handles, legend_labels, loc='best')
    return ax


def _plot_roc_curve(roc_dict, figure=None):
    figure = figure = figure if figure is not None else plt.figure(figsize=(9, 9), dpi=200, tight_layout=True)
    fpr = roc_dict['fpr']
    tpr = roc_dict['tpr']
    roc_auc = roc_dict['roc_auc']
    pos_label = roc_dict['pos_label']

    ax = plt.axes()
    roc_plot = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="BinaryModel Pipeline", pos_label=pos_label)
    roc_plot.plot(ax)
    x = np.linspace(*ax.get_xlim())
    ax.plot(x, x, 'k--', label='(AUC = 0.5)')
    ax.set_title("ROC Curve")
    ax.legend()
    return ax


def _plot_calibration_curve(calibration_dict, figure=None):
    figure = figure if figure is not None else plt.figure(figsize=(9, 9), dpi=200, tight_layout=True)
    colors = plt.cm.get_cmap("Dark2")
    ax_calibration_curve = figure.add_subplot(2, 1, 1)

    display = CalibrationDisplay(**calibration_dict, estimator_name="pipeline")
    display.plot(ax=ax_calibration_curve)
    ax_calibration_curve.grid()
    ax_calibration_curve.set_title("Calibration plot")

    ax = figure.add_subplot(2, 1, 2)

    ax.hist(
        display.y_prob,
        range=(0, 1),
        bins=10,
        label="pipeline",
        color=colors(0),
    )
    ax.set(title="Pipeline", xlabel="Mean predicted probability", ylabel="Count")
    return ax

def _plot_score_distribution(score_dict, figure=None):
    figure = figure if figure is not None else plt.figure(figsize=(9, 9), dpi=200, tight_layout=True)
    colors = plt.cm.get_cmap("Dark2")
    ax = plt.axes()
    bin_range = None
    if np.all(score_dict['scores'] <= 1.1):
        bin_range = (0, 1)
    elif np.all(score_dict['scores'] <= 101):
        bin_range = (0, 100)

    ax.hist(
        score_dict['scores'],
        range=bin_range,
        bins=10,
        label="Output Scores",
        color=colors(0),
    )
    ax.set(title="Output Scores", xlabel="Scores", ylabel="Count")
    return ax


def _make_importance_dict(permutation_importance_result):
    importance_dict = {}
    for metric, result_dict in permutation_importance_result.items():
        importance = np.clip(result_dict['importances_mean'], 0, None)
        importance_dict[metric] = {'mean': result_dict['importances_mean'],
                                                       'std': result_dict['importances_std'],
                                                       'raw': result_dict['importances'],
                                                       'relative': importance / importance.sum()}
    return importance_dict


def _plot_importances(importances_dict, feature_names, max_features=10, figure=None):
    num_plots = len(importances_dict)
    figure = figure if figure is not None else plt.figure(figsize=(num_plots*max_features, 9), dpi=200, tight_layout=True)
    palette = sns.color_palette("light:#2d6b57", as_cmap=False)
    palette = ['#2d6b57', '#357d65', '#3c8f74', '#44a182', '#4cb391', '#5dba9c', '#6fc2a7', '#81c9b2', '#93d1bd',
               '#a5d9c8', '#b7e0d3', '#c9e8de', '#dbefe9', '#edf7f4', '#ffffff'][0:10]

    palette = ['#5dba9c', '#6fc2a7', '#81c9b2', '#93d1bd',
               '#a5d9c8', '#b7e0d3', '#c9e8de', '#dbefe9', '#edf7f4']
    #palette = sns.color_palette("#c9e8de:dark", as_cmap=False)
    axs = []
    for i, (metric, metric_importances) in enumerate(importances_dict.items()):
        ax = figure.add_subplot(num_plots, 1, i+1)
        #ax.yaxis.set_label_position("right")
        #ax.yaxis.tick_right()
        raw_imp = metric_importances['raw']
        sorted_order = np.argsort(np.clip(metric_importances['mean'], 0, None))[::-1][:max_features]
        imp_plot = pd.DataFrame(raw_imp.T[:, sorted_order], columns=feature_names[sorted_order])

        # Convert from wide to long format
        imp_plot = pd.melt(imp_plot, var_name="feature_name", value_name="importance")
        avg_importances = imp_plot.groupby('feature_name').mean().sort_values(by="importance")
        if len(avg_importances) > 1:
            palette_map = pd.qcut(avg_importances['importance'], len(palette), labels=False, duplicates="drop")
            colors = np.array(palette)[palette_map.values]
        else:
            colors = np.array([palette[0]])
        # Add mean importance as hue
        imp_plot = imp_plot.merge(imp_plot.groupby('feature_name').mean(), left_on='feature_name', right_index=True, suffixes=("", "_hue"))
        imp_plot['importance_hue'] = pd.qcut(imp_plot['importance_hue'], len(palette), labels=False, duplicates="drop")
        #colors = np.array(palette)[imp_plot['importance_hue'].values]
        sns.violinplot(data=imp_plot, y="feature_name", x="importance", palette=colors,
                       inner="points", orient="h", ax=ax, scale='width')
        #ax.legend().remove()
        ax.set(title=f"{metric.title()} Importances", xlabel=f"Average {metric.title()} Change", ylabel="Feature Name")
        axs += [ax]


def _make_score_thresh_dict(y_true, y_score, thresh_decimals=3, pos_label=1):
    tot_pos = y_true.sum()
    tot_neg = len(y_true) - tot_pos
    fpr, tpr, thresh = get_binary_clf_curve(y_true, y_score, thresh_decimals=3, pos_label=pos_label)

    tp = (tpr * tot_pos)
    fn = (1 - tpr) * tot_pos
    fp = fpr * tot_neg
    tn = (1 - fpr) * tot_neg

    accuracy = (tp + tn) / (tot_pos + tot_neg)
    balanced_accuracy = 0.5 * ((tp / tot_pos) + (tn / tot_neg))
    recall = tp / tot_pos
    precision = tp / (tp + fp)
    f1 = 2 * (recall * precision) / (recall + precision)
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return {
        'tot_pos': tot_pos,
        'tot_neg': tot_neg,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'thresh': thresh,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'mcc': mcc
    }


def _plot_score_thresh(score_thresh_dict, figure=None,
                       plot_metrics=('accuracy', 'balanced_accuracy', 'recall', 'precision', 'f1', 'mcc')):
    figure = figure if figure is not None else plt.figure(figsize=(9, 9), dpi=200, tight_layout=True)

    for metric in plot_metrics:
        if metric not in score_thresh_dict:
            raise ValueError(f"{metric} not found in `score_thresh_dict`")


    num_metrics = len(plot_metrics)
    # palette = sns.color_palette("#c9e8de:dark", as_cmap=False)
    axs = []
    for i, metric in enumerate(plot_metrics):
        ax = figure.add_subplot(int(np.ceil(num_metrics / 2)), 2, i + 1)
        ax.plot(score_thresh_dict['thresh'], score_thresh_dict[metric])
        # ax.legend().remove()
        metric_formatted = metric.replace('_', ' ').title()
        ax.set(title=f"{metric_formatted} By Score Threshold",
               xlabel="Score Threshold", ylabel=f"{metric_formatted}")
        axs += [ax]

    return axs


def get_binary_clf_curve(y_true, y_score, pos_label=1, sample_weight=None, thresh_decimals=3):
    """
    Adapted from scikit-learn ranking functions:
         Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
                  Mathieu Blondel <mathieu@mblondel.org>
                  Olivier Grisel <olivier.grisel@ensta.org>
                  Arnaud Joly <a.joly@ulg.ac.be>
                  Jochen Wersdorfer <jochen@wersdoerfer.de>
                  Lars Buitinck
                  Joel Nothman <joel.nothman@gmail.com>
                  Noel Dawe <noel@dawe.me>
                  Michal Karbownik <michakarbownik@gmail.com>
         License: BSD 3 clause
    """

    # Filter out zero-weighted samples, as they should not impact the result
    if sample_weight is not None:
        nonzero_weight_mask = sample_weight != 0
        y_true = y_true[nonzero_weight_mask]
        y_score = y_score[nonzero_weight_mask]
        sample_weight = sample_weight[nonzero_weight_mask]

    # make y_true a boolean vector
    y_true = y_true == pos_label

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.0

    tot_pos = y_true.sum()
    tot_neg = y_true.size - tot_pos

    # Create equally spaced thresholds between the min and max score. We add/subtract a small value to the max/min score
    # to get the range of thresholds. This small value ensures that all scores are contained between the endpoints, even
    # when rounding.
    epsilon = 10 ** (-thresh_decimals)
    max_score = max(y_score) + epsilon
    min_score = max(0, min(y_score) - epsilon)  # hard cutoff at 0
    thresh = np.linspace(min_score, max_score, y_score.size)

    # Since the number of thresholds we need has to be the same size as y_score, this creates very small steps
    # between thresholds for large datasets. To address this, we round to the thousandths place and
    # eliminate duplicate thresholds.
    thresh = np.round(thresh, thresh_decimals)
    thresh = thresh[::-1]  # Reverse so that thresholds are decreasing

    # Here we extract indices associated with thresholds that have distinct values
    # concatenate a value for the end of the curve since np.diff returns n-1 values.
    distinct_value_indices = np.where(np.diff(thresh))[0]
    threshold_idxs = np.concatenate((distinct_value_indices, np.array([y_true.size - 1])))

    # Bucket scores into bins and count. Bin cutoffs will be unique threshold values. np.histogram requires monotonically
    # increasing cutoffs, so we need to reverse the order of thresholds
    counts, bin_cutoffs = np.histogram(y_score, thresh[threshold_idxs][::-1])

    # Need to add an extra bin since np.histogram makes k-1 bins
    counts = np.concatenate((counts, np.array([0])))

    # Now need to reverse again to get back to original order
    counts = counts[::-1]
    score_above_thresh = np.cumsum(counts)

    # score_above_thresh now corresponds to (y_score > thresh[threshold_idxs][i])) for i in range(0, threshold_idxs.size)
    # We want tps (true positives) to correspond to ((y_score > thresh[threshold_idxs][i]) & (y_true)).sum() for all i
    # We want fps (false positives) to correspond to ((y_score > thresh[threshold_idxs][i]) & (~y_true)).sum() for all i
    # If we take the cumulative sum of y_true which has already been sorted by decreasing score, then we get an array of total positives indexed by number of scores. We can then use score_above_thresh to get the number of scores above each threshold and use it as the index. The only issue is we need to add a 0 to the beginning of cumulative positives to account for zero indexing.

    # accumulate the positives with decreasing threshold
    tps = np.cumsum([0] + list(y_true * weight))[score_above_thresh]

    # accumulate the negatives with decreasing threshold
    fps = np.cumsum([0] + list((1 - y_true) * weight))[score_above_thresh]

    tpr = np.clip(tps / tot_pos, 0, 1)
    fpr = np.clip(fps / tot_neg, 0, 1)

    return fpr, tpr, thresh[threshold_idxs]
    # return fpr, tpr, score_above_thresh, pos, neg, thresh, threshold_idxs

def plot_gains_chart(y_true, y_score, figure=None):
    return _plot_lift_chart(_make_lift_dict(y_true, y_score), figure)


def plot_roc_curve(y_true, y_score, pos_label=None, figure=None):
    return _plot_roc_curve(_make_roc_dict(y_true, y_score, pos_label), figure)


def save_pipeline_visualization(pipeline, path):
    from sklearn.utils import estimator_html_repr
    from sklearn import set_config
    set_config(display='diagram')

    if not path.endswith('.html'):
        path = os.path.join(path, '.html')

    with open(path, 'w') as f:
        f.write(estimator_html_repr(pipeline))
