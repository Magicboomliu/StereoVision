from cv2 import GC_EVAL
import numpy as np
import cv2
from matplotlib import pyplot as plt


# Semi-global macthing
def opencv_sgm(imgL,imgR,window_size=5,min_disp=0,max_disp=64,SAD_block_size=3,
                disp12MaxDiffer=1,uniquenessRatio=5,vis=False):

    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    numDisparities = max_disp,
    blockSize = SAD_block_size,
    P1 = 8 * 3 * window_size**2,
    P2 = 32 * 3 * window_size**2,
    disp12MaxDiff = disp12MaxDiffer,
    uniquenessRatio = uniquenessRatio,
    speckleWindowSize = 50,
    speckleRange = 16
    )
    disparity = stereo.compute(imgL, imgR)
    if vis==True:
        disparity = disparity.astype(np.float32) / 16.0

    return disparity

# Blocking matching
def opencv_bm(left_data,right_data,numDisparities=64, blockSize=5):
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
    disparity = stereo.compute(left_data,right_data)
    return disparity




if __name__=="__main__":
    
    left_image = "test/MiddleBurry/2005/Dolls/view1.png"
    right_image = "test/MiddleBurry/2005/Dolls/view5.png"
    left_data = cv2.imread(left_image,0)
    right_data = cv2.imread(right_image,0)
    disp = opencv_bm(left_data,right_data)
    
    plt.imshow(disp,cmap='gray')
    plt.show()
    