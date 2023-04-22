import cv2
import numpy as np
import os

def img_processing(img_path):

    '''
    Image processing function.

    Input
    ------
        * img_path : dataset path.

    Return
    ------
        * processed : processed 2-channel image
        * diff      : homography parameterized by point differences.

    '''
    

    WIDTH      = 320    # image width
    HEIGHT     = 240    # image height
    PATCH_SIZE = 128    # patch size 
    RHO        = 32     # disturbance range

    # Read and resize grayscale image
    img = cv2.imread( img_path , 0)
    img = cv2.resize( img , (WIDTH, HEIGHT) )
    img = (img - 127.5)/127.5
    
    # Create a random square in the image
    points    = np.array( [ [ RHO, RHO ]                         ,
                          [ PATCH_SIZE + RHO, RHO ]              ,
                          [ RHO, PATCH_SIZE + RHO ]              ,
                          [ PATCH_SIZE + RHO, PATCH_SIZE + RHO ] ] )
    a = np.random.randint( low  = 1, high = WIDTH - 2 * RHO - PATCH_SIZE )
    b = np.random.randint( low  = 1, high = HEIGHT - 2 * RHO - PATCH_SIZE )
    points   += np.array( [[a,b]] )
    
    # Randomly perturb the corners within the range [-rho, rho] 
    moved_points = points + np.random.randint( low  = - RHO ,
                                               high = RHO   ,
                                               size = (4,2) )
    
    # Compute homography
    H = cv2.getPerspectiveTransform( points.astype('float32')       ,
                                     moved_points.astype('float32') )
    
    # Compute inverse
    H_inverse = np.linalg.inv(H)

    # Apply H to image
    warped_image = cv2.warpPerspective(img, H_inverse, (WIDTH, HEIGHT) )

    I1 = img[ points[0,1]:points[2,1] , points[0,0]:points[1,0] ]
    I2 = warped_image[ points[0,1]:points[2,1] , points[0,0]:points[1,0] ]

    # H parameterized by differences
    diff = (moved_points - points).reshape(8)

    return np.stack( [I1, I2], axis = 2 ), diff