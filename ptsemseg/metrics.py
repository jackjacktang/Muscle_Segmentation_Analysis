# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np


class runningScore(object):
    def __init__(self, n_classes, n=1):
        self.n_classes = n_classes
        self.n = n

        # 00: TP, 01: FP, 10:FN, 11:TN
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.precision = 0.0
        self.recall = 0.0

        self.TP = 0.0
        self.FP = 0.0
        self.FN = 0.0

        self.pix_auc = 0.0

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        # for lt, lp in zip(label_trues, label_preds):
        self.confusion_matrix += self._fast_hist(label_trues.flatten(), label_preds.flatten(), self.n_classes)
        self.pix_auc += (1 - np.sum(np.absolute(label_trues-label_preds)) / (label_trues.shape[0] * label_trues.shape[1]))

    # def compute_pix_auc(self, gt, pred):
    #     pix_auc = np.sum(np.absolute(gt-pred)) / (gt.shape[0] * gt.shape[1])
    #     return pix_auc
    #

    # def compute_confusion(self, gt, pred):
    #     TP = np.sum(pred[gt == 1])
    #     FP = np.sum(pred[gt == 0])
    #     FN = np.sum(gt[pred == 0])
    #
    #     if TP + FP == 0:
    #         precision = 0
    #     else:
    #         precision = TP / (TP + FP)
    #     if TP + FN == 0:
    #         recall = 0
    #     else:
    #         recall = TP / (TP + FN)
    #         # f1 = (2 * (recall * precision)) / (recall + precision)
    #     if np.isnan(precision):
    #         precision = 0
    #     if np.isnan(recall):
    #         recall = 0
    #     # print('precision: {}, recall: {}'.format(precision, recall))
    #     # return precision, recall, TP, FN, FP
    #     self.precision += precision
    #     self.recall += recall
    #     self.TP += TP
    #     self.FP += FP
    #     self.FN += FN

        return

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        # acc_cls = np.diag(hist) / hist.sum(axis=1)
        # acc_cls = np.nanmean(acc_cls)
        # iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        # mean_iu = np.nanmean(iu)

        tn = hist[0][0]
        fp = hist[0][1]
        tp = hist[1][1]
        fn = hist[1][0]

        freq = hist.sum(axis=1) / hist.sum()
        iu = tp / (tp + fp + fn)
        dice = 2 * tp / (tp + fp + tp + fn)
        # fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = {1: tp/(tp+fn), 0: tn/(tn+fp)}

        # pix_auc = self.pix_auc / self.n

        return (
            {
                # "Overall Acc: \t": acc,
                # "Mean Acc : \t": acc_cls,
                # "FreqW Acc : \t": fwavacc,
                # "Pix Acc: \t": pix_auc,
                # "IoU : \t": iu,
                "Dice : \t": dice,
                "Precision: \t": tp/(tp+fp),
                "Recall: \t": tp/(tp+fn),
                "Confusion: \t": hist,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
