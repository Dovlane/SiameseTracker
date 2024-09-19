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
from training.crops_train import crop_and_resize as crop_and_resize
from training.labels import create_BCELogit_loss_label as BCELoss
from torch.utils.data import DataLoader

class RunningAverage():
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total/float(self.steps)

class IncompatibleFolderStructure(Exception):
    pass

class IncompatibleImagenetStructure(IncompatibleFolderStructure):
    def __init__(self, msg=None):
        info = ("\n"
                "The given root directory does not conform with the expected "
                "folder structure. It should have the following structure:\n"
                "data\n"
                "├── airplane"
                "├──── airplane-1\n"
                "├          └── img\n"
                "├          └── grountruth.txt\n"
                "├──── airplane-2\n"
                "├          └── img\n"
                "├          └── grountruth.txt\n"
                "...\n"
                "├──── airplane-20\n"
                "├          └── img\n"
                "├          └── grountruth.txt\n"
                "├── bicycle"
                "├──── bicycle-1\n"
                "├          └── img\n"
                "├          └── grountruth.txt\n"
                "...\n"
                "├──── bicycle-20\n"
                "├          └── img\n"
                "├          └── grountruth.txt\n")
        if msg is not None:
            info = info + msg
        super().__init__(info)


def check_folder_tree(base_path):
    global classes
    subclasses_counts = {}
    for parent in classes:
        parent_path = os.path.join(base_path, parent)
        if not os.path.isdir(parent_path):
            print(f"Missing or invalid parent folder: {parent_path}")
            return None
        
        subclasses_count = len(os.listdir(parent_path))
        subclasses_counts[parent] = subclasses_count
        for i in range(1, subclasses_count + 1):
            subclass_folder = os.path.join(parent_path, f"{parent}-{i}")
            img_folder = os.path.join(subclass_folder, "img")
            groundtruth_file = os.path.join(subclass_folder, "groundtruth.txt")
            if not os.path.isdir(subclass_folder):
                print(f"Missing or invalid folder: {parent}\\{parent}-{i}")
                return None
            if not os.path.isdir(img_folder):
                print(f"Missing or invalid folder: {parent}\\{parent}-{i}\\img")
                return None
            if not os.path.isfile(groundtruth_file):
                print(f"Missing or invalid folder: {parent}\\{parent}-{i}\\groundtruth.txt")
                return None
    return subclasses_counts


classes = ["airplane", "bicycle"]


class ImageLASOT_train(Dataset):
    def __init__(self, imagenet_dir, 
                 reference_size=127, search_size=255, final_size=33,
                 label_fcn=BCELoss, upscale_factor=4,
                 max_frame_sep=50, pos_thr=25, neg_thr=50,
                 cxt_margin=0.5, single_label=True): 
        self.subclasses_indexes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        self.init(imagenet_dir, reference_size, search_size, final_size, label_fcn, upscale_factor, max_frame_sep, pos_thr, neg_thr, cxt_margin, single_label)
        

    def init(self, imagenet_dir,
        reference_size=127, search_size=255, final_size=33,
                 label_fcn=BCELoss, upscale_factor=4,
                 max_frame_sep=50, pos_thr=25, neg_thr=50,
                 cxt_margin=0.5, single_label=True):
        self.start_subclass_idx = self.subclasses_indexes[0]
        self.classes = classes
        self.classes_num = len(self.classes)
        self.subclasses_counts = check_folder_tree(imagenet_dir)
        if self.subclasses_counts is None:
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
        self.label_fcn = label_fcn
        if single_label:
            self.label = self.label_fcn(self.final_size, self.pos_thr,
                                        self.neg_thr,
                                        upscale_factor=self.upscale_factor)
        else:
            self.label = None
        
        self.load_annotations_and_frames()
        self.list_pairs = self.build_test_pairs()
        self.self_len = 0
        self.set_list_idx()


    def set_list_idx(self):
        self.list_idx = {}
        for class_type, large_dir_img_paths in zip(self.frames.keys(), self.frames.values()): 
            self.list_idx[class_type] = []
            i = 0
            for dir_img_paths in large_dir_img_paths:
                for dir_img_path in dir_img_paths:
                    self.list_idx[class_type].append(i)
                self.self_len += len(dir_img_paths)
                i += 1 

    def set_root_dirs(self, root):
        self.root = root
        self.dir_img = {}
        self.groundtruth_path_map = {}
        for class_type, subclasses_count in zip(self.subclasses_counts.keys(), self.subclasses_counts.values()):
            self.dir_img[class_type] = []
            self.groundtruth_path_map[class_type] = []
            for i in self.subclasses_indexes:
                subclass_folder = os.path.join(root, class_type, f"{class_type}-{i}")
                self.dir_img[class_type].append(os.path.join(subclass_folder, "img"))
                groundtruth_path = os.path.join(subclass_folder, "groundtruth.txt")
                self.groundtruth_path_map[class_type].append(groundtruth_path)

    def build_test_pairs(self):
        random.seed(100)
        list_pairs = []

        for class_idx, class_name in enumerate(self.frames):  
            for seq_idx, frame_seq in enumerate(self.frames[class_name]):
                for frame_idx, frame in enumerate(frame_seq):
                    list_pairs.append([class_idx, seq_idx, *self.get_pair(class_idx, seq_idx, frame_idx)])
        random.shuffle(list_pairs)
        random.seed()
        return list_pairs


    def __len__(self):
        return self.self_len


    def get_pair(self, class_idx, seq_idx, frame_idx = None):
        if (seq_idx + self.start_subclass_idx) not in self.subclasses_indexes:
            raise Exception("Sorry I still have only one folder in my database!")

        class_name = self.classes[class_idx]
        size = len(self.frames[class_name][seq_idx])
        pad = 20
        max_allowed_frame = max(1, size - self.max_frame_sep - pad)
        if frame_idx is None:
            first_frame_idx = random.randint(0, max_allowed_frame)
        else:
            first_frame_idx = min(frame_idx, max_allowed_frame)
        
        min_frame_idx = max(0, (first_frame_idx - self.max_frame_sep))
        max_frame_idx = min(max(1, size - pad), (first_frame_idx + self.max_frame_sep))
        second_frame_idx = random.randint(min_frame_idx, max_frame_idx)

        return first_frame_idx, second_frame_idx
        

    def ref_context_size(self, h, w):
        margin_size = self.cxt_margin*(w + h)
        ref_size = sqrt((w + margin_size) * (h + margin_size))
        ref_size = (int(ref_size)//2)*2 + 1
        return ref_size


    def load_annotations_and_frames(self):
        groundtruth_cols = ["x", "y", "w", "h"]
        annotation_cols = ["x1", "y1", "x2", "y2"]
        groundtruth = 'groundtruth.txt'
        groundtruth_map = {}
        image_paths = {}
        for class_type, groundtruth_paths, dir_img_paths in zip(self.groundtruth_path_map.keys(), self.groundtruth_path_map.values(), self.dir_img.values()):
            groundtruth_map[class_type] = []
            image_paths[class_type] = []
            for i in self.subclasses_indexes:
                data = pd.read_csv(groundtruth_paths[i - self.start_subclass_idx], sep=',', header=None, names=groundtruth_cols)
                images = glob.glob(os.path.join(dir_img_paths[i - self.start_subclass_idx], '*'))
                df = pd.DataFrame(data)
                df["x1"] = df["x"]
                df["y1"] = df["y"]
                df.drop(columns=["x", "y"], inplace = True)
                df["x2"] = df["x1"] + df["w"]
                df["y2"] = df["y1"] + df["h"]
                invalid_indicies =  df[(df['h'] <= 10) | (df['w'] <= 10)].index
                df.drop(invalid_indicies)
                images = [image_path for idx, image_path in enumerate(images) if idx not in invalid_indicies]
                df.drop(columns=["w", "h"], inplace = True)
                groundtruth_map[class_type].append(df)
                image_paths[class_type].append(images)

        self.frames = image_paths
        self.annotations = groundtruth_map
    

    def preprocess_sample(self, class_idx, seq_idx, first_idx, second_idx): 
        class_name = self.classes[class_idx]
        reference_frame_path = self.frames[class_name][seq_idx][first_idx]
        search_frame_path = self.frames[class_name][seq_idx][second_idx]
        try:
            ref_annot = self.annotations[class_name][seq_idx].iloc[first_idx]
        except IndexError as e:
            print('len(self.frames[class_name][seq_idx]) =', len(self.frames[class_name][seq_idx]))
            print('first_idx =', first_idx)
            print('seq_idx =', seq_idx)
            print('class_idx =', class_idx)
            print(f"IndexError caught: {e}")
        
        srch_annot = self.annotations[class_name][seq_idx].iloc[second_idx]

        ref_x1, ref_x2 = ref_annot['x1'].astype(int), ref_annot['x2'].astype(int)
        ref_y1, ref_y2 = ref_annot['y1'].astype(int), ref_annot['y2'].astype(int)
        ref_w = ref_x2 - ref_x1
        ref_h = ref_y2 - ref_y1

        ref_frame = crop_and_resize(reference_frame_path, ref_x1, ref_y1, ref_w, ref_h, 127)

        srch_x1, srch_x2 = srch_annot['x1'].astype(int), srch_annot['x2'].astype(int)
        srch_y1, srch_y2 = srch_annot['y1'].astype(int), srch_annot['y2'].astype(int)
        srch_w = srch_x2 - srch_x1
        srch_h = srch_y2 - srch_y1

        srch_frame = crop_and_resize(search_frame_path, srch_x1, srch_y1, srch_w, srch_h, 255)

        if self.label is not None:
            label = self.label
        else:
            label = self.label_fcn(self.final_size, self.pos_thr, self.neg_thr,
                                   upscale_factor=self.upscale_factor)

        out_dict = {'ref_frame': ref_frame, 'srch_frame': srch_frame,
                    'label': label, 'class_idx' : class_idx, 'seq_idx' : seq_idx, 
                    'ref_idx': first_idx, 'srch_idx': second_idx}
        return out_dict     

    def __getitem__(self, idx):
        item = self.preprocess_sample(*self.list_pairs[idx])
        return item



class ImageLASOT_val(ImageLASOT_train):
    def __init__(self, *args, **kwargs):
        self.subclasses_indexes = [16, 17, 18]
        super().init(*args)

    def set_root_dirs(self, root):
        super().set_root_dirs(root)

    def get_frames(self):
        return super().get_frames()

    def build_test_pairs(self):
        return super().build_test_pairs()

    def __getitem__(self, idx):
        return super().__getitem__(idx)


class ImageLASOT_test(ImageLASOT_train):
    def __init__(self, *args, **kwargs):
        self.subclasses_indexes = [19, 20]
        super().init(*args)

    def set_root_dirs(self, root):
        super().set_root_dirs(root)

    def get_frames(self):
        return super().get_frames()

    def build_test_pairs(self):
        return super().build_test_pairs()

    def __getitem__(self, idx):
        return super().__getitem__(idx)


imageLASOT = ImageLASOT_val('data/')

out_dict = imageLASOT[600]
print('len(imageLASOT)', len(imageLASOT))
print('out_dict[class_idx]', out_dict['class_idx'])
print('out_dict[seq_idx]', out_dict['seq_idx'])
print('out_dict[ref_idx]', out_dict['ref_idx'])
print('out_dict[srch_idx]', out_dict['srch_idx'])


train_dataloader = DataLoader(imageLASOT, batch_size=8, shuffle = True)
for i, data in enumerate(train_dataloader):
    ref_frame_tensor, srch_frame_tensor, label = data['ref_frame'], data['srch_frame'], data['label']
    print(ref_frame_tensor.shape)
    print(srch_frame_tensor.shape)
    print(label.shape)
    if i == 100:
        break