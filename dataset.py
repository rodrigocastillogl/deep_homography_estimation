# stdlib modules
import glob
from typing import 
import os

# third party modules
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataloader, Dataset
from torchvision.transforms import ToTensor
import torch.nn.functional as F


class HomographyDataset(Dataset):
    """
    Deep homography estimation dataset
    Attributes
    ----------
        * base_dir: path to folder containing dataset (images).
        * width: image width.
        * height: image height.
        * patch_size: patch size.
        * rho: disturbance range.
        * __img_paths: list of the filepath to every iage in the dataset.
    Methods
    -------
        * __init__, __len__, __get_random_square,
    """

    def __init__( self, base_dir,width: int = 320 , height: int = 240 ,
                  patch_size: int = 128 , rho: int = 32 ):
        
        """
        Constructor method
        Input
        -----
            * Class attributes: base_dir, width, height, patch_size, rho.
        Output
        ------
            None
        """

        # save parameters
        self.width = width
        self.height = height
        self.patch_size = patch_size
        self.rho = rho
        self.base_dir = base_dir
        self.__img_paths = glob.glob(os.path.join(base_dir, "*"))


    def __len__(self) -> int:
        """
        Dataset length method
        Computes the length of the list of image filepaths.
        Input
        -----
            None
        Output
        ------
            Number of images inside of dataset directory
        """

        return len(self.__img_paths)


    def __get_random_square( self ) -> np.ndarray:
        
        """
        Compute random perturbed square within image and rho boundaries.
        Input
        -----
            None
        Output
        ------
            points: (4, 2)-shape numpy ndarray containing perturbed corners.
        """
        # define corners' relative positions
        points = np.array([ [self.rho, self.rho],
                            [self.patch_size + self.rho, self.rho],
                            [self.rho, self.patch_size + self.rho],
                            [self.patch_size + self.rho, self.patch_size + self.rho]])
        
        # compute corner's absolute positions
        a = np.random.randint(low=1, high = self.width - 2 * self.rho - self.patch_size)
        b = np.random.randint(low=1, high = self.height - 2 * self.rho - self.patch_size)
        points += np.array([[a, b]])

        return points
    

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        """
        Access dataset element using an integer index. Reads image from img_path
        at position idx from __img_paths attribute, performs preprocessing steps
        and returns two image cropped patches and their corresponding homography H.
        Input
        -----
            * idx: integer-valued indexing value
        Output
        ------
            imgs: tensor containing both cropped patches from original and warped images.
            H: (8,)-shape tensor containing the point-difference representation
               of the Homography.
            """
        
        # read and resize grayscale image
        img = cv2.imread(__img_paths[idx], 0)
        img = cv2.resize(img, (self.width, self.height))
        img = (img - 127.5) / 127.5 # normalize

        # create a random square within image and rho constraints
        points = self.__get_random_square()
        
        # perturb randomly the corners within the range [-rho, rho]
        moved_points = points + np.random.randint( low = -self.rho ,
                                                   high = self.rho ,
                                                   size = (4, 2)   )

        # compute homography between points and moved_points
        H = cv2.getPerspectiveTransform( points.astype('float32')       ,
                                         moved_points.astype('float32') )
        
        # compute inverse
        inv_H = np.linalg.inv(H)

        # apply H to image
        warped_image = cv2.warpPerspective( img, H_inverse, (self.width, self.height) )

        # trim images bassed on previously computed random square
        sub_img1 = img[ points[0,1]:points[2,1], points[0,0]:points[1,0] ]
        sub_img2 = warped_img[ points[0,1]:points[2,1], points[0,0]:points[1,0] ]

        # convert to tensor
        sub_img1 = torch.from_numpy(sub_img1)
        sub_img2 = torch.from_numpy(sub_img2)

        # stack images
        imgs = torch.stack( [sub_img1, sub_img2], dim = 2 )

        # parameterize H using point differences
        H = (moved_points - points).reshape(8)
        H = torch.from_numpy(H)

        return imgs, H
