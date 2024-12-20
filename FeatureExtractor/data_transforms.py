import pandas as pd
from PIL import Image
import numpy as np
import cv2

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

'''
A set of callable data transforms, along with a Dataset class
DatasetWrapper class efficiently handles creation of DataLoaders

This code is copied or slightly modified from:
https://github.com/vkola-lab/tmi2022/tree/main/feature_extractor/data_aug
'''

# create a dataset class
class DataSet_CSV():
    '''
    Dataset class to use with torch DataLoader class.
    Reads a csv file containing a filepath to an image in each line.
    '''
    def __init__(self, csv_file, transform=None):
        self.files_list = pd.read_csv(csv_file)
        self.transform = transform
    def __len__(self):
        return len(self.files_list)
    def __getitem__(self, idx):
        temp_path = self.files_list.iloc[idx, 0]
        img = Image.open(temp_path)
        img = transforms.functional.to_tensor(img)
        if self.transform:
            sample = self.transform(img)
        return sample
    

# define a series of transforms to be applied to the dataset
class ToPIL(object):
    '''
    Converts torch tensor to PIL image.
    Technically redundant, but included to allow Dataset_CSV.__get_item__ to return a torch tensor
    with or without transforms included
    '''
    def __call__(self, sample):
        img = sample
        img = transforms.functional.to_pil_image(img)
        return img

class ColorJitter(object):
    '''
    Randomly changes contrast, brightness, saturation, and hue according to tunable parameter s
    '''
    def __init__(self, s=1.0):
        self.s = s
        self.color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)

    def __call__(self, img):
        # update the ColorJitter instance with the current value of s
        self.color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        return self.color_jitter(img)

class GaussianBlur(object):
    '''
    randomly applies a Gaussian blur to image with 50% chance
    '''
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
#            print(self.kernel_size)
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample
    

# define a callable class that applies transforms twice to the same sample
class FeatExtractDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi/255, xj/255 # normalize images
    

# combine these transforms into a pipeline and prepare the data for training
class DataSetWrapper(object):
    '''
    Class to handle applying transforms to Dataset_CSV, splitting into train/validation sets,
    and returning respective DataLoaders for training a model with this data
    '''
    def __init__(self, batch_size, num_workers, valid_size, input_shape, s, data_path):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.s = s
        self.input_shape = eval(input_shape)
        self.data_path = data_path

    def get_data_loaders(self):
        data_augment = self._data_transform_pipeline_()
        train_dataset = DataSet_CSV(csv_file=self.data_path, transform=FeatExtractDataTransform(data_augment))
        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def _data_transform_pipeline_(self):
        # get a set of data augmentation transformations 
        # in addition to the classes defined above, randomly flip the image, and randomly convert to grayscale

        data_transforms = transforms.Compose([ToPIL(),
                                      transforms.Resize((self.input_shape[0],self.input_shape[1])),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomApply([ColorJitter()], p=0.8),
                                      transforms.RandomGrayscale(p=0.2),
                                      GaussianBlur(kernel_size=int(0.06 * 128)),
                                      transforms.ToTensor()])
        return data_transforms

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)
        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)
        return train_loader, valid_loader
