import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
from pathlib import Path
import os


sns.set_theme()

def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec

# def plot_pr_curve(px, py, ap, save_dir=Path('pr_curve.png'), names=()):
#     # Precision-recall curve
#     fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
#     py = np.stack(py, axis=1)

#     # if 0 < len(names) < 21:  # display per-class legend if < 21 classes
#     #     for i, y in enumerate(py.T):
#     #         ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
#     # else:
#     #     ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

#     ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
#     ax.set_xlabel('Recall')
#     ax.set_ylabel('Precision')
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
#     ax.set_title('Precision-Recall Curve')
#     fig.savefig(save_dir, dpi=250)
#     plt.close(fig)

def plot_pr_curve(px, py, ap, save_dir=Path('pr_curve.png'), names=()):
    
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title('Precision-Recall Curve')
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    

def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16, prefix=""):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict
    return px, py, ap
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / f'{prefix}PR_curve.png', names)
        # plot_mc_curve(px, f1, Path(save_dir) / f'{prefix}F1_curve.png', names, ylabel='F1')
        # plot_mc_curve(px, p, Path(save_dir) / f'{prefix}P_curve.png', names, ylabel='Precision')
        # plot_mc_curve(px, r, Path(save_dir) / f'{prefix}R_curve.png', names, ylabel='Recall')

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)

stats0 = np.load('/home/dtpthao/workspace/yolov5/real.npy', allow_pickle=True)
stats1 = np.load('/home/dtpthao/workspace/yolov5/blur_10.npy', allow_pickle=True)
stats2 = np.load('/home/dtpthao/workspace/yolov5/blur_20.npy', allow_pickle=True)
stats3 = np.load('/home/dtpthao/workspace/yolov5/blur_30.npy', allow_pickle=True)
stats4 = np.load('/home/dtpthao/workspace/yolov5/blur_40.npy', allow_pickle=True)
stats = [stats1, stats2, stats3, stats4, stats0]

names={0: 'L_Vocal Fold', 1: 'L_Arytenoid cartilage', 2: 'Benign lesion', 3: 'Malignant lesion', 4: 'R_Vocal Fold', 5: 'R_Arytenoid cartilage'}
color = ['darkgray', 'gray', 'black', 'red', 'blue']
fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
for ix, st in enumerate(stats):
    save_dir = "/home/dtpthao/workspace/yolov5"
    px, py, ap = ap_per_class(*st, plot=True, save_dir=save_dir, names=names)
    py = np.stack(py, axis=1)
    if ix == 4:
        # label = 'Original (non-blur) {:.3f} mAP@0.5'.format(ap[:, 0].mean())
        label = 'Original (non-blur)'
        ax.plot(px, py.mean(1), linewidth=4, color=color[ix], label=label)
        # print("MEAL mAP50-95:", ap.mean(1).mean())
    elif ix == 3:
        # label = 'blur {}0% {:.3f} mAP@0.5'.format(ix, ap[:, 0].mean())
        label = 'blur {}0%'.format(ix+1)
        ax.plot(px, py.mean(1), linewidth=4, color=color[ix], label=label)
    else:
        # label = 'blur {}0% {:.3f} mAP@0.5'.format(ix+1, ap[:, 0].mean())
        label = 'blur {}0%'.format(ix+1)
        ax.plot(px, py.mean(1), linewidth=2, color=color[ix], label=label)
        # print("blur {} mAP50-95:".format(ix), ap.mean(1).mean())
    

ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
# ax.set_title('Precision-Recall Curve')
fig.savefig(os.path.join(save_dir, "PR_curve.png"), dpi=300)
plt.close(fig)