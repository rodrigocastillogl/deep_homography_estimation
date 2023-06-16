import glob
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class HomographyDataset(Dataset):
    """
    Deep homography estimation dataset.
    Attributes
    ----------
        * path: path to folder containing dataset (images).
        * width: image width.
        * height: image height.
        * patch_size: patch size.
        * rho: disturbance range.
        * image_paths: list of the filepath to every image in the dataset.
    Methods
    -------
        * __init__, __len__, random_square
    """

    def __init__( self, path, width = 320 , height = 240,
                  patch_size = 128 , rho = 32 ):
        
        """
        Constructor.
        Input
        -----
            * Class attributes: path, width, height, patch_size, rho.
        Output
        ------
            None
        """

        # save parameters
        self.width = width
        self.height = height
        self.patch_size = patch_size
        self.rho = rho
        self.path = path
        
        # get image paths
        self.image_paths = glob.glob( os.path.join(path, '*') )


    def __len__(self):
        """
        Dataset length method (number of images).
        Input
        -----
            None
        Output
        ------
            Number of images in the dataset directory.
        """

        return len( self.image_paths )


    def random_square( self ):
        """
        Compute random perturbed square within image.
        Input
        -----
            None
        Output
        ------
            points: (4, 2)-shape numpy ndarray containing perturbed corners.
        """
        
        # create a rondom square
        points = np.array( [ [self.rho, self.rho],
                             [self.patch_size + self.rho, self.rho],
                             [self.rho, self.patch_size + self.rho],
                             [self.patch_size + self.rho, self.patch_size + self.rho]])
        a = np.random.randint(low=1, high = self.width - 2 * self.rho - self.patch_size)
        b = np.random.randint(low=1, high = self.height - 2 * self.rho - self.patch_size)
        points += np.array( [[a, b]] )

        return points
    

    def __getitem__(self, idx):
        
        """
        Access dataset element using an integer index, performs preprocessing
        steps and returns two image cropped patches and their corresponding homography.
        Input
        -----
            * idx: integer-valued indexing value.
        Output
        ------
            imgs: tensor containing both cropped patches from original and warped images.
            point_diff: (8,)-shape tensor containing the point-difference representation
                        of the Homography.
        """
        
        # read, re-scale and normalize grayscale image
        img = cv2.imread( self.image_paths[idx], 0)
        img = cv2.resize( img, (self.width, self.height) )
        img = ( img - 127.5 ) / 127.5 

        # create a random square within image and rho constraints
        points = self.random_square()
        
        # perturb randomly the corners within the range [-rho, rho]
        moved_points = points + np.random.randint( low = -self.rho ,
                                                   high = self.rho ,
                                                   size = (4, 2)   )

        # compute homography between points and moved_points
        H = cv2.getPerspectiveTransform( points.astype('float32')       ,
                                         moved_points.astype('float32') )
        
        # compute inverse
        H_inverse = np.linalg.inv(H)

        # apply H_inverse to image
        warped_img = cv2.warpPerspective( img, H_inverse, (self.width, self.height) )

        # trim images bassed on previously computed random square
        sub_img1 = img[ points[0,1]:points[2,1], points[0,0]:points[1,0] ]
        sub_img2 = warped_img[ points[0,1]:points[2,1], points[0,0]:points[1,0] ]

        # convert to tensor
        sub_img1 = torch.from_numpy(sub_img1)
        sub_img2 = torch.from_numpy(sub_img2)

        # stack images
        imgs = torch.stack( [sub_img1, sub_img2], dim = 2 )

        # parameterize H using point differences
        point_diff = (moved_points - points).reshape(8)
        point_diff = torch.from_numpy(H)

        return imgs, point_diff