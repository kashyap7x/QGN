import os
import json
from pathlib import Path
from scipy.misc import imread


train_13_txt = Path('./data/SUNRGBD_train13.txt')
test_13_txt = Path('./data/SUNRGBD_test13.txt')
train_37_txt = Path('./data/SUNRGBD_train37.txt')
test_37_txt = Path('./data/SUNRGBD_test37.txt')

train_13_odgt = Path('./data/train_13_SUNRGBD.odgt')
test_13_odgt = Path('./data/test_13_SUNRGBD.odgt')
train_37_odgt = Path('./data/train_37_SUNRGBD.odgt')
test_37_odgt = Path('./data/test_37_SUNRGBD.odgt')


def make_SUNRGBD(txt, odgt):
    with txt.open(mode='r') as fi:
        with odgt.open(mode='w+') as fo:
            lines = [line.rstrip('\n').split(' ') for line in fi]
            for l in lines:
                img = imread('./data/'+l[0], mode='RGB')
                item = {"width": img.shape[1], "fpath_img": l[0], "height": img.shape[0], "fpath_segm": l[2]}
                fo.write(f'{json.dumps(item)}\n')


def main():
    make_SUNRGBD(train_13_txt, train_13_odgt)
    make_SUNRGBD(test_13_txt, test_13_odgt)
    make_SUNRGBD(train_37_txt, train_37_odgt)
    make_SUNRGBD(test_37_txt, test_37_odgt)


if __name__ == '__main__':
    main()
