
import os

import os.path as osp 
import numpy as np 

from glob import glob 


class CityscapesDatset: 
    
    """
    Cityscapes dataset.
    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.

    Code Reference : 
        [1] https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/datasets/cityscapes.py
    """

    CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]

    def __init__(self, data_dir, test_mode = False):

        self.data_dir = data_dir
        self.img_dir = osp.join(data_dir, 'leftImg8bit_trainvaltest/leftImg8bit')
        self.ann_dir = osp.join(data_dir, 'gtFine_trainvaltest/gtFine')
        self.img_suffix = '_leftImg8bit.png'
        self.seg_map_suffix = '_gtFine_labelIds.png'

        

    def load_annotations(self): 

        """Load annotation from directory.
        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            
        Returns:
            list[dict]: All image info of dataset.

        Code Reference : 
            [1] https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/datasets/cityscapes.py   
        """
        img_infos = []
        img_list = []
        for _, _, files in os.walk(self.img_dir):
            for file in files:
                print(file)
                if file.endswith(self.img_suffix):

                    img_list.append(file)


        for img in img_list:
            img_info = dict(filename=img)
            seg_map = img.replace(self.img_suffix, self.seg_map_suffix)
            img_info['ann'] = dict(seg_map=seg_map)
            img_infos.append(img_info)

        return img_infos

    def prepare_train_img(self, idx): 

        return None

    def prepare_test_img(self, idx): 

        return None

    def __getitem__(self, idx): 
        
        return None 

    





def main(): 
    data_dir = '/home/sss/UOS-SSaS Dropbox/05. Data/00. Benchmarks/01. cityscapes'
    cityscapes_dataset = CityscapesDatset(data_dir)
    img_infos = cityscapes_dataset.load_annotations()
    print(img_infos)
    return None 

if __name__ == "__main__" : 
    main()