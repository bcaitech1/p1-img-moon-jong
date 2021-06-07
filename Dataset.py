from torch.utils.data import Dataset, DataLoader
from PIL import Image

import os
import pandas as pd
from glob import glob

import numpy as np

"""

"""


class SeperateLabel:
    mask = 0
    incorrect = 6
    not_wear = 12


class ThreeHot:
    sex = [0, 1]

    age = [2, 3, 4]

    mask = [5, 6, 7]


def hot_index(age, gender):
    """
    Args:
        age(int): given by ImageOnHotDataset which decide which element of hot_list(at ImageOnHotDataset.listdir())
                  make hot(1)
        gender(int): given by ImageOnHotDataset which decide which element of hot_list(at ImageOnHotDataset.listdir())
                  make hot(1)

    return:
        age_idx: int
        gender_idx: int
    """
    if age < 30:
        age_idx = 0
    elif age >= 30 and age < 60:
        age_idx = 1
    elif age >= 60:
        age_idx = 2

    gender_idx = 0 if gender == 'male' else 1
    return age_idx, gender_idx


def labeling(age, gender):
    """
    Args:
        age:
        gender:
    """
    if age < 30:
        age_label = 0
    elif age >= 30 and age < 60:
        age_label = 1
    elif age >= 60:
        age_label = 2

    gender = 0 if gender == 'male' else 3
    return age_label + gender


def csv_labeling(csv):
    """
    Args:
        csv:
    """
    if csv.age < 30:
        age_label = 0
    elif csv.age >= 30 and csv.age < 58:
        age_label = 1
    elif csv.age >= 58:
        age_label = 2

    gender = 0 if csv.gender == 'male' else 3

    return age_label + gender


class TrainValidSplit:
    def __init__(self, seed):
        """
        Args:
            seed: to set random seed
        """
        np.random.seed(seed)

    def sampling(self, target_csv):
        """
        Args:
            target_csv(pd.DataFrame): DataFrame which is not seperated by train and valid set

        return:
            train_df, valid_df (pd.DataFrame)
            it seperates the set 8:2(train:valid) ratio
        """
        target = target_csv
        target['label'] = target_csv.apply(csv_labeling, axis=1)
        train_dict = {}
        valid_dict = {}
        sorted_label = sorted(target['label'].value_counts().index)

        train_df = pd.DataFrame()
        valid_df = pd.DataFrame()

        for label in sorted_label:
            total_label = target[target['label'] == label]
            random_choiced = np.random.choice(range(len(total_label.index)), len(total_label), replace=False)
            shuffled_df = total_label.iloc[random_choiced]
            sample_ratio = int(len(random_choiced) * 0.8)

            train_dict[label] = shuffled_df[:sample_ratio]
            valid_dict[label] = shuffled_df[sample_ratio:]

        for item in train_dict.values():
            train_df = train_df.append(item)

        for item in valid_dict.values():
            valid_df = valid_df.append(item)

        return train_df, valid_df


class ImageDataset(Dataset):
    _unnecesarry = ['id', 'race']

    def __init__(self, csv_path=None, image_path='./input/data/train', activate_transform=True, train=True,
                 transform=None, extra_path=None):
        """
        Args:
            csv_path(str): csv path, which provided by competition. It should contrains Imagepath, gender, age, id, race

            image_path(str): image_path, which contrains directories that have the 7 images (incorrect, 5 correct, not_wear)

            activate_transform(bool): parameter to activate transform if activate_transform= False then it do not transform images
                                even though transform is given

            train(bool): by this parameter this Dataset is decided that it will be used by train set  or valid set

            transform(transform.Compose): process of transforms

            extra_path: extra csv path which have extra image informations
        """
        self.activate_transform = activate_transform
        if csv_path is not None:
            self.train_csv = pd.read_csv(csv_path)
        self.train = train
        self.transform = transform
        self.image_path = image_path
        if extra_path is not None:
            self.extra_data = pd.read_csv(extra_path)
        else:
            self.extra_data = None

        if self.train:
            for un_ele in self._unnecesarry:
                if un_ele in self.train_csv.columns:
                    self.train_csv = self.train_csv.drop(un_ele, axis=1)
            self._train_img_path = os.path.join(self.image_path, 'images')
        else:
            self._test_img_path = os.path.join(self.image_path, 'images')
        self.image_label = []
        self.list_dir()

    def __getitem__(self, idx):
        if self.train:
            image, label = self.image_label[idx]
            image = Image.open(image)

            if self.activate_transform:
                image = self.transform(image)
            return image, label

        else:
            image = self.image_label[idx]
            image = Image.open(image)

            if self.activate_transform:
                image = self.transform(image)

            return image

    def __len__(self):
        return len(self.image_label)

    def list_dir(self):
        """
        no Args
        no return

        this function make a list like [[img, label], [img, label]...] form
        label's form is integer by class (0 - 17)

        """


        if self.train:
            for gender, age, img_path in self.train_csv.values:
                image_lists = glob(os.path.join(self._train_img_path, img_path, '*'))
                pre_label = labeling(age, gender)
                for image in image_lists:
                    if 'incorrect' in image:
                        self.image_label.append([image, SeperateLabel.incorrect + pre_label])
                    elif 'normal' in image:
                        self.image_label.append([image, SeperateLabel.not_wear + pre_label])
                    else:
                        self.image_label.append([image, SeperateLabel.mask + pre_label])

            if self.extra_data is not None:
                img_path = os.path.join(self.image_path, 'part1')

                for gender, age, path in self.extra_data.values:
                    self.image_label.append([os.path.join(img_path, path),
                                             SeperateLabel.not_wear + labeling(age, gender)])




        else:
            self.image_label = glob(os.path.join(self._test_img_path, '*'))


class ImageOnHotDataset(Dataset):
    _unnecesarry = ['id', 'race']


    def __init__(self, csv_path=None, image_path='./input/data/train', activate_transform=True, train=True,
                 transform=None, extra_path=None):
        """
        Args:
            csv_path(str): csv path, which provided by competition. It should contrains Imagepath, gender, age, id, race

            image_path(str): image_path, which contrains directories that have the 7 images (incorrect, 5 correct, not_wear)

            activate_transform(bool): parameter to activate transform if activate_transform= False then it do not transform images
                                even though transform is given

            train(bool): by this parameter this Dataset is decided that it will be used by train set  or valid set

            transform(transform.Compose): process of transforms

            extra_path: extra csv path which have extra image informations
        """
        self.activate_transform = activate_transform
        if csv_path is not None:
            self.train_csv = pd.read_csv(csv_path)
        self.train = train
        self.transform = transform
        self.image_path = image_path
        if extra_path is not None:
            self.extra_data = pd.read_csv(extra_path)
        else:
            self.extra_data = None

        if self.train:
            for un_ele in self._unnecesarry:
                if un_ele in self.train_csv.columns:
                    self.train_csv = self.train_csv.drop(un_ele, axis=1)
            self._train_img_path = os.path.join(self.image_path, 'images')
        else:
            self._test_img_path = os.path.join(self.image_path, 'images')
        self.image_label = []
        self.list_dir()

    def __getitem__(self, idx):

        if self.train:
            image, label = self.image_label[idx]
            image = Image.open(image)

            if self.activate_transform:
                image = self.transform(image)
            return image, label

        else:
            image = self.image_label[idx]
            image = Image.open(image)

            if self.activate_transform:
                image = self.transform(image)

            return image

    def __len__(self):
        return len(self.image_label)

    def list_dir(self):
        """
                no Args
                no return

                this function make a list like [[img, label], [img, label]...] form
                label's form is like 3-hot encoding.
                0,1 idx : gender (male, female)
                2,3,4 idx: age (under30, between 30 and 60, over 60)
                5,6,7 idx: mask (correct, incorrect, normal)

                """
        if self.train:
            for gender, age, img_path in self.train_csv.values:
                image_lists = glob(os.path.join(self._train_img_path, img_path, '*'))
                for image in image_lists:
                    hot_list = np.array([0, 0, 0, 0, 0, 0, 0, 0])
                    a, g = hot_index(age, gender)
                    if 'incorrect' in image:
                        hot_list[[ThreeHot.sex[g], ThreeHot.age[a], ThreeHot.mask[1]]] = 1
                        self.image_label.append([image, hot_list])
                    elif 'normal' in image:
                        hot_list[[ThreeHot.sex[g], ThreeHot.age[a], ThreeHot.mask[2]]] = 1
                        self.image_label.append([image, hot_list])
                    else:
                        hot_list[[ThreeHot.sex[g], ThreeHot.age[a], ThreeHot.mask[0]]] = 1
                        self.image_label.append([image, hot_list])

            if self.extra_data is not None:
                """
                extra data have only not_wear asian age between 0 - 90
                so it's maks label is only normal[0, 0, 1]
                """
                img_path = os.path.join(self.image_path, 'part1')

                for gender, age, path in self.extra_data.values:
                    hot_list = np.array([0, 0, 0, 0, 0, 0, 0, 0])
                    a, g = hot_index(age, gender)
                    hot_list[[ThreeHot.sex[g], ThreeHot.age[a], ThreeHot.mask[1]]] = 1
                    self.image_label.append([os.path.join(img_path, path), hot_list])




        else:
            self.image_label = glob(os.path.join(self._test_img_path, '*'))
