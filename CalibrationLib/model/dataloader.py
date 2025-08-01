import copy
import os
import sys
import glob
import json
import torch
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as f

from torchvision.transforms import v2
from torch.utils.data import Dataset
from PIL import Image

from utils.utils_keypoints import KeypointsDB
from utils.utils_keypointsWC import KeypointsWCDB


class SoccerNetCalibrationDataset(Dataset):

    def __init__(self, root_dir, split, transform, main_cam_only=True):

        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.match_info = json.load(open(root_dir + split + '/match_info.json'))
        self.files = self.get_image_files(rate=1)

        if main_cam_only:
            self.get_main_camera()

    def get_image_files(self, rate=3):
        files = glob.glob(os.path.join(self.root_dir + self.split, "*.jpg"))
        files.sort()
        if rate > 1:
            files = files[::rate]
        return files


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.files[idx]
        image = Image.open(img_name)
        data = json.load(open(img_name.split('.')[0] + ".json"))
        data = self.correct_labels(data)
        sample = self.transform({'image': image, 'data': data})

        img_db = KeypointsDB(sample['data'], sample['image'])
        target, mask = img_db.get_tensor_w_mask()
        image = sample['image']

        return image, torch.from_numpy(target).float(), torch.from_numpy(mask).float()


    def get_main_camera(self):
        self.files = [file for file in self.files if int(self.match_info[file.split('/')[-1]]['ms_time']) == \
                      int(self.match_info[file.split('/')[-1]]['replay_time'])]

    def correct_labels(self, data):
        if 'Goal left post left' in data.keys():
            data['Goal left post left '] = copy.deepcopy(data['Goal left post left'])
            del data['Goal left post left']

        return data


class WorldCup2014Dataset(Dataset):

    def __init__(self, root_dir, split, transform):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        assert self.split in ['train_val', 'test'], f'unknown dataset type {self.split}'

        self.files = glob.glob(os.path.join(self.root_dir + self.split, "*.jpg"))
        self.homographies = glob.glob(os.path.join(self.root_dir + self.split, "*.homographyMatrix"))
        self.num_samples = len(self.files)

        self.files.sort()
        self.homographies.sort()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.get_image_by_index(idx)
        homography = self.get_homography_by_index(idx)
        img_db = KeypointsWCDB(image, homography, (960,540))
        target, mask = img_db.get_tensor_w_mask()

        sample = self.transform({'image': image, 'target': target, 'mask': mask})

        return sample['image'], sample['target'], sample['mask']

    def convert_homography_WC14GT_to_SN(self, H):
        T = np.eye(3)
        #T[0, -1] = -115 / 2
        #T[1, -1] = -74 / 2
        yard2meter = 0.9144
        S = np.eye(3)
        S[0, 0] = yard2meter
        S[1, 1] = yard2meter
        H_SN = S @ (T @ H)

        return H_SN

    def get_image_by_index(self, index):
        img_file = self.files[index]
        image = Image.open(img_file)
        return image

    def get_homography_by_index(self, index):
        homography_file = self.homographies[index]
        with open(homography_file, 'r') as file:
            lines = file.readlines()
            matrix_elements = []
            for line in lines:
                matrix_elements.extend([float(element) for element in line.split()])
        homography = np.array(matrix_elements).reshape((3, 3))
        homography = self.convert_homography_WC14GT_to_SN(homography)
        homography = torch.from_numpy(homography)
        homography = homography / homography[2:3, 2:3]
        return homography


class TSWorldCupDataset(Dataset):

    def __init__(self, root_dir, split, transform):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        assert self.split in ['train', 'test'], f'unknown dataset type {self.split}'

        self.files_txt = self.get_txt()

        self.files = self.get_jpg_files()
        self.homographies = self.get_homographies()
        self.num_samples = len(self.files)

        self.files.sort()
        self.homographies.sort()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.get_image_by_index(idx)
        homography = self.get_homography_by_index(idx)
        img_db = KeypointsWCDB(image, homography, (960,540))
        target, mask = img_db.get_tensor_w_mask()

        sample = self.transform({'image': image, 'target': target, 'mask': mask})

        return sample['image'], sample['target'], sample['mask']


    def get_txt(self):
        with open(self.root_dir + self.split + '.txt', 'r') as file:
            lines = file.readlines()
        lines = [line.strip() for line in lines]
        return lines

    def get_jpg_files(self):
        all_jpg_files = []
        for dir in self.files_txt:
            full_dir = self.root_dir + "Dataset/80_95/" + dir
            jpg_files = []
            for file in os.listdir(full_dir):
                if file.lower().endswith('.jpg') or file.lower().endswith('.jpeg'):
                    jpg_files.append(os.path.join(full_dir, file))

            all_jpg_files.extend(jpg_files)

        return all_jpg_files

    def get_homographies(self):
        all_homographies = []
        for dir in self.files_txt:
            full_dir = self.root_dir + "Annotations/80_95/" + dir
            homographies = []
            for file in os.listdir(full_dir):
                if file.lower().endswith('.npy'):
                    homographies.append(os.path.join(full_dir, file))

            all_homographies.extend(homographies)

        return all_homographies


    def convert_homography_WC14GT_to_SN(self, H):
        T = np.eye(3)
        #T[0, -1] = -115 / 2
        #T[1, -1] = -74 / 2
        yard2meter = 0.9144
        S = np.eye(3)
        S[0, 0] = yard2meter
        S[1, 1] = yard2meter
        H_SN = S @ (T @ H)

        return H_SN

    def get_image_by_index(self, index):
        img_file = self.files[index]
        image = Image.open(img_file)
        return image

    def get_homography_by_index(self, index):
        homography_file = self.homographies[index]
        homography = np.load(homography_file)
        homography = self.convert_homography_WC14GT_to_SN(homography)
        homography = torch.from_numpy(homography)
        homography = homography / homography[2:3, 2:3]
        return homography
