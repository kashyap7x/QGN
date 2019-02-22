import os
import json
import torch
import lib.utils.data as torchdata
import cv2
from torchvision import transforms
from scipy.misc import imread, imresize
import numpy as np
from data_utils.gen_labels import dense2quad


# Round x to the nearest multiple of p and x' >= x
def round2nearest_multiple(x, p):
    return ((x - 1) // p + 1) * p


class TrainDataset(torchdata.Dataset):
    def __init__(self, odgt, opt, max_sample=-1, batch_per_gpu=1, quadtree_levels=6):
        self.root_dataset = opt.root_dataset
        self.imgSize = opt.imgSize
        self.imgMaxSize = opt.imgMaxSize
        self.random_flip = opt.random_flip
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant
        self.batch_per_gpu = batch_per_gpu
        self.quadtree_levels = quadtree_levels        
        self.transform_dict = None
        self.crop = None

        if opt.transform_dict:
            self.transform_dict = json.loads(opt.transform_dict)

        if opt.cropSize > 0:
            self.crop = opt.cropSize

        # classify images into two classes: 1. h > w and 2. h <= w
        self.batch_record_list = [[], []]

        # override dataset length when trainig with batch_per_gpu > 1
        self.cur_idx = 0

        # mean and std
        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.])
            ])

        self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        self.if_shuffled = False
        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def _get_sub_batch(self):
        while True:
            # get a sample record
            this_sample = self.list_sample[self.cur_idx]
            if this_sample['height'] > this_sample['width']:
                self.batch_record_list[0].append(this_sample) # h > w, go to 1st class
            else:
                self.batch_record_list[1].append(this_sample) # h <= w, go to 2nd class

            # update current sample pointer
            self.cur_idx += 1
            if self.cur_idx >= self.num_sample:
                self.cur_idx = 0
                np.random.shuffle(self.list_sample)

            if len(self.batch_record_list[0]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[0]
                self.batch_record_list[0] = []
                break
            elif len(self.batch_record_list[1]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[1]
                self.batch_record_list[1] = []
                break
        return batch_records

    def __getitem__(self, index):
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        if not self.if_shuffled:
            np.random.shuffle(self.list_sample)
            self.if_shuffled = True

        # get sub-batch candidates
        batch_records = self._get_sub_batch()

        # resize all images' short edges to the chosen size
        if isinstance(self.imgSize, list):
            this_short_size = np.random.choice(self.imgSize)
        else:
            this_short_size = self.imgSize

        # calculate the BATCH's height and width
        # since we concat more than one samples, the batch's h and w shall be larger than EACH sample
        batch_resized_size = np.zeros((self.batch_per_gpu, 2), np.int32)
        for i in range(self.batch_per_gpu):
            img_height, img_width = batch_records[i]['height'], batch_records[i]['width']
            this_scale = min(this_short_size / min(img_height, img_width), \
                    self.imgMaxSize / max(img_height, img_width))
            img_resized_height, img_resized_width = img_height * this_scale, img_width * this_scale
            batch_resized_size[i, :] = img_resized_height, img_resized_width
        batch_resized_height = np.max(batch_resized_size[:, 0])
        batch_resized_width = np.max(batch_resized_size[:, 1])

        # Here we must pad both input image and segmentation map to size h' and w' so that p | h' and p | w'
        batch_resized_height = int(round2nearest_multiple(batch_resized_height, self.padding_constant))
        batch_resized_width = int(round2nearest_multiple(batch_resized_width, self.padding_constant))

        batch_images = torch.zeros(self.batch_per_gpu, 3, batch_resized_height, batch_resized_width)
        batch_segms = []
        for l in range(self.quadtree_levels):
            batch_segms.append(torch.zeros(self.batch_per_gpu, batch_resized_height // (2**l), \
                                batch_resized_width // 2**l).long())

        for i in range(self.batch_per_gpu):
            this_record = batch_records[i]

            # load image and label
            image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
            segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])
            img = imread(image_path, mode='RGB')
            segm = imread(segm_path)

            assert(img.ndim == 3)
            assert(segm.ndim == 2)
            assert(img.shape[0] == segm.shape[0])
            assert(img.shape[1] == segm.shape[1])

            if self.random_flip == True:
                random_flip = np.random.choice([0, 1])
                if random_flip == 1:
                    img = cv2.flip(img, 1)
                    segm = cv2.flip(segm, 1)

            # note that each sample within a mini batch has different scale param
            img = imresize(img, (batch_resized_size[i, 0], batch_resized_size[i, 1]), interp='bilinear')
            segm = imresize(segm, (batch_resized_size[i, 0], batch_resized_size[i, 1]), interp='nearest')

            if self.crop:
                x1 = np.random.randint(batch_resized_size[i, 1] - self.crop)
                y1 = np.random.randint(batch_resized_size[i, 0] - self.crop)
                img = img[y1: y1 + self.crop, x1: x1 + self.crop, :]
                segm = segm[y1: y1 + self.crop, x1: x1 + self.crop]

             # image to float
            img = img.astype(np.float32)[:, :, ::-1] # RGB to BGR!!!
            img = img.transpose((2, 0, 1))
            img = self.img_transform(torch.from_numpy(img.copy()))
            
            # add to batch
            batch_images[i][:, :img.shape[1], :img.shape[2]] = img
            
            # to avoid seg label misalignment
            segm_rounded_height = round2nearest_multiple(segm.shape[0], self.padding_constant)
            segm_rounded_width = round2nearest_multiple(segm.shape[1], self.padding_constant)
            segm_rounded = np.zeros((segm_rounded_height, segm_rounded_width), dtype='uint8')
            segm_rounded[:segm.shape[0], :segm.shape[1]] = segm

            seg_copy = segm_rounded.copy().astype(np.int)

            if self.transform_dict:
                for k, v in self.transform_dict.items():
                    seg_copy[segm == int(k)] = v
                    segm = seg_copy
            else:
                segm = seg_copy - 1 # label from -1 to 149
            
            # convert to quadtree
            seg_copy = segm + 1
            quadtree = dense2quad(seg_copy, self.quadtree_levels)
            
            for l in range(self.quadtree_levels):                        
                segm = quadtree[self.quadtree_levels-l]
                seg_copy = segm.copy().astype(np.int)
                seg_copy[segm == 0] = -1
                seg_copy[segm == -1] = 0
                segm = seg_copy
                batch_segms[l][i][:segm.shape[0], :segm.shape[1]] = torch.from_numpy(segm.astype(np.int)).long()

        output = dict()
        output['img_data'] = batch_images
        for l in range(self.quadtree_levels):
            output['seg_label_'+str(l)] = batch_segms[l]

        return output

    def __len__(self):
        return int(1e6) # It's a fake length due to the trick that every loader maintains its own list
        #return self.num_sampleclass


class ValDataset(torchdata.Dataset):
    def __init__(self, odgt, opt, max_sample=-1, start_idx=-1, end_idx=-1):
        self.root_dataset = opt.root_dataset
        self.imgSize = opt.imgSize
        self.imgMaxSize = opt.imgMaxSize
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant
        if opt.transform_dict:
            self.transform_dict = json.loads(opt.transform_dict)
        else:
            self.transform_dict = None

        # mean and std
        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.])
            ])

        self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]

        if start_idx >= 0 and end_idx >= 0: # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image and label
        image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
        segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])
        img = imread(image_path, mode='RGB')
        img = img[:, :, ::-1] # BGR to RGB!!!
        segm = imread(segm_path)

        ori_height, ori_width, _ = img.shape

        img_resized_list = []
        for this_short_size in self.imgSize:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                    self.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_height = round2nearest_multiple(target_height, self.padding_constant)
            target_width = round2nearest_multiple(target_width, self.padding_constant)

            # resize
            img_resized = cv2.resize(img.copy(), (target_width, target_height))

            # image to float
            img_resized = img_resized.astype(np.float32)
            img_resized = img_resized.transpose((2, 0, 1))
            img_resized = self.img_transform(torch.from_numpy(img_resized))

            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)

        seg_copy = segm.copy().astype(np.int)

        if self.transform_dict:
            for k, v in self.transform_dict.items():
                seg_copy[segm == int(k)] = v
                segm = seg_copy
        else:
            segm = seg_copy - 1 # label from -1 to 149

        segm = torch.from_numpy(segm.astype(np.int)).long()
        batch_segms = torch.unsqueeze(segm, 0)

        output = dict()
        output['img_ori'] = img.copy()
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        output['seg_label'] = batch_segms.contiguous()
        output['info'] = this_record['fpath_img']
        return output

    def __len__(self):
        return self.num_sample


class TestDataset(torchdata.Dataset):
    def __init__(self, odgt, opt, max_sample=-1):
        self.imgSize = opt.imgSize
        self.imgMaxSize = opt.imgMaxSize
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant

        # mean and std
        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.])
            ])

        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image and label
        image_path = this_record['fpath_img']
        img = imread(image_path, mode='RGB')
        img = img[:, :, ::-1] # BGR to RGB!!!

        ori_height, ori_width, _ = img.shape

        img_resized_list = []
        for this_short_size in self.imgSize:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                    self.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_height = round2nearest_multiple(target_height, self.padding_constant)
            target_width = round2nearest_multiple(target_width, self.padding_constant)

            # resize
            img_resized = cv2.resize(img.copy(), (target_width, target_height))

            # image to float
            img_resized = img_resized.astype(np.float32)
            img_resized = img_resized.transpose((2, 0, 1))
            img_resized = self.img_transform(torch.from_numpy(img_resized))

            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)

        # segm = torch.from_numpy(segm.astype(np.int)).long()

        # batch_segms = torch.unsqueeze(segm, 0)

        # batch_segms = batch_segms - 1 # label from -1 to 149
        output = dict()
        output['img_ori'] = img.copy()
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        # output['seg_label'] = batch_segms.contiguous()
        output['info'] = this_record['fpath_img']
        return output

    def __len__(self):
        return self.num_sample
