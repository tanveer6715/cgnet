
import os
import cv2 
import random

import os.path as osp 
import numpy as np 
import cityscapesscripts.helpers.labels as CSLabels # to be deprecated 

from tqdm import tqdm
from glob import glob 

CLASSES = ('road', 'sidewalk', 'building', 'wall', 
            'fence', 'pole', 'traffic light', 'traffic sign',
            'vegetation', 'terrain', 'sky', 'person', 
            'rider', 'car', 'truck', 'bus', 
            'train', 'motorcycle', 'bicycle')

PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
            [0, 80, 100], [0, 0, 230], [119, 11, 32]]



class CityscapesDatset: 
    
    """
    Cityscapes dataset.
    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.

    Code Reference : 
        [1] https://github.com/open-mmlab/mmsegmentation
    """


    def __init__(self, data_dir, data_type = 'train'):

        self.classes = CLASSES
        self.palette = PALETTE
        self.data_dir = data_dir
        self.img_dir = osp.join(data_dir, 'leftImg8bit_trainvaltest/leftImg8bit', data_type)
        self.ann_dir = osp.join(data_dir, 'gtFine_trainvaltest/gtFine', data_type)
        self.img_suffix = '_leftImg8bit.png'
        self.seg_map_suffix = '_gtFine_labelIds.png'

        # load annotations
        self.img_infos = self.load_img_infos()

        
    def load_img_infos(self): 

        """Load annotation from directory.
        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            
        Returns:
            list[dict]: All image info of dataset.

        Code Reference : 
            [1] https://github.com/open-mmlab/mmsegmentation
        """
        img_infos = []
        img_list = []

        for _, _, files in os.walk(self.img_dir):
            for file in files:
                if file.endswith(self.img_suffix):
                    img_list.append(file)


        for img in img_list:
            img_info = dict(filename=img)
            seg_map = img.replace(self.img_suffix, self.seg_map_suffix)
            img_info['ann'] = dict(seg_map=seg_map)
            img_infos.append(img_info)

        return img_infos

    def prepare_img(self, idx): 

        """ Read image from the dataset directory
        Args:
                        
        Returns:
            
        """

        img_filename = self.img_infos[idx]['filename']
        img_prefix = img_filename.split('_')[0]

        img_path = osp.join(self.img_dir, img_prefix, img_filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def prepare_seg_mask(self, idx): 

        """ Read segmentation mask from the annotation directory
        Args:
                        
        Returns:
            
        """

        seg_filename = self.img_infos[idx]['ann']['seg_map']
        seg_prefix = seg_filename.split('_')[0]

        seg_path = osp.join(self.ann_dir, seg_prefix, seg_filename)
        seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)

        return CityscapesDatset._convert_to_label_id(seg)

    @staticmethod
    def _convert_to_label_id(seg):
        """Convert trainId to id for cityscapes."""
        seg_copy = seg.copy()
        for label in CSLabels.labels:
            # print(label.name)
            seg_copy[seg == label.id] = label.trainId
        return seg_copy

    @staticmethod
    def _get_class_weight(data_dir):
        """get class weight of cityscapes dataset 
        
        
        """
        cityscapes_dataset = CityscapesDatset(data_dir)
        img_infos = cityscapes_dataset.img_infos

        data_length = len(cityscapes_dataset)
        
        hist_sum = np.zeros(20)
        num_pxls_present = np.zeros(20)

        for idx in tqdm(range(100)): 
            data = cityscapes_dataset[idx]
            segmentation_mask = data['segmentation_mask']
            hist, _ = np.histogram(segmentation_mask, bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 ,13, 14, 15, 16, 17, 18, 19, 20])
            hist_sum = hist_sum + hist
            class_existence_map = np.array(hist, dtype = bool)
            num_of_pxls = segmentation_mask.shape[0]*segmentation_mask.shape[1]
            num_pxls_present = num_pxls_present + class_existence_map*num_of_pxls

        np.seterr(divide='ignore', invalid='ignore')
        freq = np.divide(hist_sum, num_pxls_present)

        freq = np.nan_to_num(freq)
        if 'median' in mode :
            divider = np.nanmedian(freq)
        elif mode == 'mean' :
            divider = np.nanmean(freq)

        #class_weight = np.divide(median_freq, freq)
        # class weight from ENet paper 
        class_weight = 1 / np.log(1.02 + (median_freq/freq))
        class_weight[-1] = 0

        np.save('class_weight_cityscapes.npy', class_weight)
            
    

    def __len__ (self) : 

        return len(self.img_infos)

    
    def __getitem__(self, idx):
        
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.
        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """
        data = {}
        # data['image'] = self.prepare_img(idx)
        # data['segmentation_mask'] = self.prepare_seg_mask(idx)

        image = self.prepare_img(idx)
        label = self.prepare_seg_mask(idx)

        """
        Code Reference : 
        https://github.com/wutianyiRosun/CGNet/blob/master/dataset/cityscapes.py
        """

        f_scale = 0.5 + random.randint(0, 15) / 10.0  #random resize between 0.5 and 2 
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)

        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        
        data['image'] = image 
        data['segmentation_mask'] = label

        return data


def test(): 
    data_dir = '/home/soojin/UOS-SSaS Dropbox/05. Data/00. Benchmarks/01. cityscapes'
    cityscapes_dataset = CityscapesDatset(data_dir)

    cityscapes_dataset._get_class_weight(data_dir)
    
    

    
    

        
    
    return None 

if __name__ == "__main__" : 
    test()