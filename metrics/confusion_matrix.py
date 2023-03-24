import numpy as np

def ConfusionMatrix(p, gt):
    p = np.round(np.array(p))
    gt = np.array(gt)
    tp = np.sum((p == 1) & (gt == 1)).astype(int)
    fp = np.sum((p == 1) & (gt == 0)).astype(int)
    tn = np.sum((p == 0) & (gt == 0)).astype(int)
    fn = np.sum((p == 0) & (gt == 1)).astype(int)
    return tp, fp, tn ,fn

def MetricsFromCM(tp, fp, tn, fn, epsilon=1.0e-6):
    tpr = (tp + epsilon)/(tp + fn + epsilon)
    fpr = (fp + epsilon)/(fp + tn + epsilon)
    precision = (tp + epsilon)/(tp + fp + epsilon)
    tnr = (tn + epsilon)/(fp + tn + epsilon)
    return tpr, fpr, precision, tnr
