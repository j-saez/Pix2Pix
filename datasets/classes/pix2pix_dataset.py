#############
## IMPORTS ##
#############

import os
import torchvision
import numpy                  as np
import albumentations         as A
import albumentations.pytorch as A_torch
from PIL                      import Image
from torch.utils.data         import Dataset
from utils.configuration      import Configuration

#######################
## globals and const ##
#######################

IMGS_CHS = 3
DATASET_PATH = f'{os.getcwd()}/datasets/data'

#############
## Classes ##
#############

class Pix2PixDataset(Dataset):

    def __init__(self, config: Configuration, split: str) -> None:
        """
        Common class to load a dataset that is splitted in train and val folders. It loads the dataset contained in dataset_path.
        Inputs:
            >> config: (Configuration) Training configuration.
        """
        if split != 'val' and split != 'train':
            raise ValueError(f'split must be val or train, not {split}.')
        self.split = split

        direction = config.dataparams.direction
        if direction != 'b_to_a' and direction != 'a_to_b':
            raise ValueError(f'Direction must be a_to_b or b_to_a, not {direction}.')

        self.direction = direction
        self.desired_size = config.hyperparams.gen_input_size
        imgs_dir = f'{DATASET_PATH}/{config.dataparams.dataset_name}/{split}'
        self.pathed_img_names = [f'{imgs_dir}/{filename}' for filename in os.listdir(imgs_dir)]
        self.imgs_chs = load_resized_images(self.pathed_img_names[0], 256)[0].shape[-1]

        train_transforms, test_transforms, denorm_tr = self.get_transforms()
        self.train_transforms = train_transforms 
        self.test_transforms = test_transforms 
        self.denorm_transforms = denorm_tr
         
        return

    def get_transforms(self):
        std  = [0.5] * self.imgs_chs
        mean = [0.5] * self.imgs_chs

        train_transforms = A.Compose(
            [
                A.Resize(286,286),
                A.RandomResizedCrop(256, 256),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=mean, std=std),
                A_torch.ToTensorV2(),  # Apply ToTensorV2 directly to both source and target
            ],
            additional_targets={"image0": "image"}
        )

        test_transforms = A.Compose(
            [
                A.Normalize(mean=mean, std=std),
                A_torch.ToTensorV2(),  # Apply ToTensorV2 directly to both source and target
            ],
            additional_targets={"image0": "image"}
        )

        denorm_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Normalize([-m/s for m, s in zip(mean, std)], [1/s for s in std]), ])

        return train_transforms, test_transforms, denorm_transforms

    def __len__(self):
        return len(self.pathed_img_names)

    def __getitem__(self, idx):
        imgA, imgB = load_resized_images(self.pathed_img_names[idx], self.desired_size)

        transformed_data = None
        if self.split == 'train':
            if self.direction == 'a_to_b': transformed_data = self.train_transforms(image=imgA, image0=imgB)
            elif self.direction == 'b_to_a': transformed_data = self.train_transforms(image=imgB, image0=imgA)

        else:
            if self.direction == 'a_to_b': transformed_data = self.test_transforms(image=imgA, image0=imgB)
            elif self.direction == 'b_to_a': transformed_data = self.test_transforms(image=imgB, image0=imgA)

        return transformed_data['image'], transformed_data['image0']

###############
## Functions ##
###############

def load_resized_images(filename, desired_size):
    """
    Load the images from the specified directory and resizes them to the desired size. It expects images os size (CHS, HEIGHT, 2*WIDTH) as there have two be two images, the source one and the target one.
    Inputs:
        >> filename: (str) Pathed filename for the file to be loaded.
        >> desired_size: (int) Desired size for squared image after resizing.
    Outputs:
        >> imgA: numpy array of size (desired_size, desired_size, chs) containing the source images.
        >> imgB: numpy array of size (desired_size, desired_size, chs) containing the target images.
    """
    full_img = np.array(Image.open(filename))
    
    orig_size = full_img.shape[0]
    desired_size = (desired_size, desired_size)
    
    imgA, imgB = full_img[:, :orig_size, :], full_img[:, orig_size:, :]
    imgA = Image.fromarray(imgA).resize(desired_size)
    imgB = Image.fromarray(imgB).resize(desired_size)
    imgA, imgB = np.array(imgA), np.array(imgB)
    
    return imgA, imgB
