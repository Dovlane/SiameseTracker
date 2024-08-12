import os
from os.path import join, relpath, isfile, abspath
from math import sqrt
import random
import glob

import numpy as np
from imageio import imread
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch
import pandas as pd


import sys
import os
from os.path import join, isdir, isfile, splitext
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training.crops_train import crop_img , resize_and_pad
from training.labels import create_BCELogit_loss_label as BCELoss

class IncompatibleFolderStructure(Exception):
    pass

class IncompatibleImagenetStructure(IncompatibleFolderStructure):
    def __init__(self, msg=None):
        info = ("\n"
                "The given root directory does not conform with the expected "
                "folder structure. It should have the following structure:\n"
                "airplane\n"
                "├── airplane_1\n"
                "├       └── img\n"
                "├       └── grountruth.txt\n"
                "├── airplane_2\n"
                "├       └── img\n"
                "├       └── grountruth.txt\n"
                "...\n"
                "├── airplane_20\n"
                "├       └── img\n"
                "├       └── grountruth.txt\n")
        if msg is not None:
            info = info + msg
        super().__init__(info)


def check_folder_tree(base_path):
    for i in range(1, 21):  # Checking for airplane_1 to airplane_20
        airplane_folder = os.path.join(base_path, f"airplane-{i}")
        img_folder = os.path.join(airplane_folder, "img")
        groundtruth_file = os.path.join(airplane_folder, "groundtruth.txt")
        
        if not os.path.isdir(airplane_folder):
            print(f"Missing or invalid folder: {airplane_folder}")
            return False
        if not os.path.isdir(img_folder):
            print(f"Missing img folder: {img_folder}")
            return False
        if not os.path.isfile(groundtruth_file):
            print(f"Missing groundtruth.txt file: {groundtruth_file}")
            return False
    
    return True



class ImageLaSOT(Dataset):
    def __init__(self, imagenet_dir, transforms=ToTensor(),
                 reference_size=127, search_size=255, final_size=33,
                 label_fcn=BCELoss, upscale_factor=4,
                 max_frame_sep=50, pos_thr=25, neg_thr=50,
                 cxt_margin=0.5, single_label=True, #img_read_fcn=imread,
                 metadata_file=None, save_metadata=None):

        if not check_folder_tree(imagenet_dir):
            raise IncompatibleImagenetStructure
        self.set_root_dirs(imagenet_dir)
        self.max_frame_sep = max_frame_sep
        self.reference_size = reference_size
        self.search_size = search_size
        self.upscale_factor = upscale_factor
        self.cxt_margin = cxt_margin
        self.final_size = final_size
        self.pos_thr = pos_thr
        self.neg_thr = neg_thr
        self.transforms = transforms
        self.label_fcn = label_fcn
        self.label_fcn = label_fcn
        if single_label:
            self.label = self.label_fcn(self.final_size, self.pos_thr,
                                        self.neg_thr,
                                        upscale_factor=self.upscale_factor)
        else:
            self.label = None
        self.frames = self.load_frame_paths()
        self.annotations = self.load_annotations()
        self.set_list_idx()

    def set_list_idx(self):
        self.list_idx = []
        rows = len(self.frames)
        for seq_idx in range(rows):
            for _ in self.frames[seq_idx]:
                self.list_idx.append(seq_idx)

    def set_root_dirs(self, root):
        self.root = root
        self.dir_img = []
        self.groundtruth_paths = []
        # self.dir_img = join(root, 'img')
        for i in range(1, 21):  # Checking for airplane_1 to airplane_20
            airplane_folder = os.path.join(root, f"airplane-{i}")
            self.dir_img.append(os.path.join(airplane_folder, "img"))
            groundtruth_path = os.path.join(airplane_folder, "groundtruth.txt")
            self.groundtruth_paths.append(groundtruth_path)

    def __len__(self):
        return len(self.frames)

    def get_scene_dirs(self): # For now, I interpreted as "get_scene_images"
        image_paths = []
        
        for i in range(1, 21):  # Checking for airplane_1 to airplane_20
            airplane_folder = os.path.join(self.root, f"airplane-{i}")
            img_folder = os.path.join(airplane_folder, "img")
            if os.path.isdir(img_folder):
                images = glob.glob(os.path.join(img_folder, '*'))
                image_paths.append(images)
                
        return image_paths
        
    def load_frame_paths(self):
        self.frames = self.get_scene_dirs()
        return self.frames

    def get_pair(self, seq_idx, frame_idx = None):
        if  not (0 <= seq_idx and seq_idx <= 19):
            raise Exception("Sorry I still have only one folder in my database!")

        size = len(self.frames[seq_idx])
        if frame_idx is None:
            first_frame_idx = random.randint(0, size-2)
        else:
            first_frame_idx = frame_idx

        min_frame_idx = max(0, (first_frame_idx - self.max_frame_sep))
        max_frame_idx = min(size - 1, (first_frame_idx + self.max_frame_sep))
        
        second_frame_idx = random.randint(min_frame_idx, max_frame_idx)

        return first_frame_idx, second_frame_idx
        

    def ref_context_size(self, h, w): # extended box for target when target
        margin_size = self.cxt_margin*(w + h)
        ref_size = sqrt((w + margin_size) * (h + margin_size))
        # make sur ref_size is an odd number
        ref_size = (int(ref_size)//2)*2 + 1
        return ref_size


    def load_annotations(self):
        groundtruth_cols = ["x", "y", "w", "h"]
        annotation_cols = ["x1", "y1", "x2", "y2"]
        groundtruth = 'groundtruth.txt'
        groundtruth_list = []
        for i in range(20):
            self.annotations = pd.read_csv(self.groundtruth_paths[i], sep=',', header=None, names=groundtruth_cols)
            df = pd.DataFrame(self.annotations)
            df["x1"] = df["x"]
            df["y1"] = df["y"]
            df.drop(columns=["x", "y"], inplace = True)
            df["x2"] = df["x1"] + df["w"]
            df["y2"] = df["y1"] + df["h"]
            df.drop(columns=["w", "h"], inplace = True)
            groundtruth_list.append(df)
        return groundtruth_list


        


    def get_frame_at(self, idx): # idx -> frame_idx; seq_idx should be added!
        return self.frames[idx]
    

    def preprocess_sample(self, seq_idx, first_idx, second_idx): # first_idx, second_idx -> first_frame_idx, second_frame_idx; seq_idx should be added!
        reference_frame_path = self.frames[seq_idx][first_idx]
        search_frame_path = self.frames[seq_idx][second_idx]
        ref_annot = self.annotations[seq_idx].iloc[first_idx]
        srch_annot = self.annotations[seq_idx].iloc[second_idx]

        
        ref_w = (ref_annot['x2'].astype(int) - ref_annot['x1'].astype(int)) / 2
        ref_h = (ref_annot['y2'].astype(int) - ref_annot['y1'].astype(int)) / 2
        ref_ctx_size = self.ref_context_size(int(ref_h), int(ref_w))
        ref_cx = (ref_annot['x2'].astype(int) + ref_annot['x1'].astype(int)) / 2
        ref_cy = (ref_annot['y2'].astype(int) + ref_annot['y1'].astype(int)) / 2

        # ref_frame = self.img_read(reference_frame_path)
        ref_frame = Image.open(reference_frame_path)
        ref_frame = np.float32(ref_frame)

        ref_frame, pad_amounts_ref = crop_img(ref_frame, ref_cy, ref_cx, ref_ctx_size)
        
                                       #resize_fcn=self.resize_fcn)
        try:
            ref_frame = resize_and_pad(ref_frame, self.reference_size, pad_amounts_ref,
                                       reg_s=ref_ctx_size, use_avg=True)
                                       #resize_fcn=self.resize_fcn)
        except AssertionError:
            print('Fail Ref: ', reference_frame_path)
            raise

        srch_ctx_size = ref_ctx_size * self.search_size / self.reference_size
        srch_ctx_size = (srch_ctx_size//2)*2 + 1

        srch_cx = (srch_annot['x2'].astype(int) + srch_annot['x1'].astype(int))/2
        srch_cy = (srch_annot['y2'].astype(int) + srch_annot['y1'].astype(int))/2


        srch_frame = Image.open(search_frame_path)
        srch_frame = np.float32(srch_frame)
        srch_frame, pad_amounts_srch = crop_img(srch_frame, srch_cy, srch_cx, srch_ctx_size)
        try:
            srch_frame = resize_and_pad(srch_frame, self.search_size, pad_amounts_srch,
                                        reg_s=srch_ctx_size, use_avg=True)
                                        #resize_fcn=self.resize_fcn)
        except AssertionError:
            print('Fail Search: ', search_frame_path)
            raise

        if self.label is not None:
            label = self.label
        else:
            label = self.label_fcn(self.final_size, self.pos_thr, self.neg_thr,
                                   upscale_factor=self.upscale_factor)
        
        ref_frame = self.transforms(ref_frame)
        srch_frame = self.transforms(srch_frame)

        out_dict = {'ref_frame': ref_frame, 'srch_frame': srch_frame,
                    'label': label, 'seq_idx' : seq_idx,'ref_idx': first_idx,
                    'srch_idx': second_idx }
        return out_dict

    def __getitem__(self, idx):
        seq_idx = self.list_idx[idx]
        first_idx, second_idx = self.get_pair(seq_idx)
        return self.preprocess_sample(seq_idx, first_idx, second_idx)        


imageLaSOT = ImageLaSOT('data/airplane/')
out_dict = imageLaSOT[600]
print(out_dict['seq_idx'])
print(out_dict['ref_frame'].shape)
print(out_dict['srch_frame'].shape)