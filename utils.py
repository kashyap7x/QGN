import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
import torch
from torch.autograd import Variable


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret


def colorEncode(labelmap, colors):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))
    return labelmap_rgb


def accuracy(batch_data, pred):
    (imgs, segs, infos) = batch_data
    _, preds = torch.max(pred.data.cpu(), dim=1)
    valid = (segs >= 0)
    acc = 1.0 * torch.sum(valid * (preds == segs)) / (torch.sum(valid) + 1e-10)
    return acc, torch.sum(valid)


def intersectionAndUnion(batch_data, pred, numClass):
    (imgs, segs, infos) = batch_data
    _, preds = torch.max(pred.data.cpu(), dim=1)

    # compute area intersection
    intersect = preds.clone()
    intersect[torch.ne(preds, segs)] = -1

    area_intersect = torch.histc(intersect.float(),
                                 bins=numClass,
                                 min=0,
                                 max=numClass-1)

    # compute area union:
    preds[torch.lt(segs, 0)] = -1
    area_pred = torch.histc(preds.float(),
                            bins=numClass,
                            min=0,
                            max=numClass-1)
    area_lab = torch.histc(segs.float(),
                           bins=numClass,
                           min=0,
                           max=numClass-1)
    area_union = area_pred + area_lab - area_intersect
    return area_intersect, area_union


def gather(a, window=(2, 2)):
    A = np.arange(torch.numel(a)).reshape(a.size())
    batch = A.shape[0]
    channels = A.shape[1]
    block = (batch, channels, window[0], window[1])
    shape = (1, 1, A.shape[2]// block[2], A.shape[3]// block[3])+ block
    strides = (block[0] * A.strides[0], block[1] * A.strides[1], block[2] * A.strides[2], block[3] * A.strides[3])+ A.strides
    indices = ast(A, shape=shape, strides=strides).T.reshape(batch, channels * block[2] * block[3], A.shape[2] // block[2], A.shape[3] // block[3])
    ind_variable = Variable(torch.from_numpy(indices)).long()
    if torch.cuda.is_available():
        ind_variable = ind_variable.cuda()
    a_gathered = torch.take(a, ind_variable)
    return a_gathered