#coding:utf-8
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from utils import unpickle, saved_to_pickle

# 设置迭代终止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


objp = np.zeros((9 * 6, 3), np.float32)  #我用的是9×6的棋盘格，可根据自己棋盘格自行修改相关参数
objp[:, :2] = np.mgrid[0:6, 0:9].T.reshape(-1, 2)

# 用arrays存储所有图片的object points 和 image points
objpoints = []  # 3d points in real world space
imgpointsR = []  # 2d points in image plane
imgpointsL = []

import os
num_images = len(os.listdir("left"))



# 本次实验采集里共计30组待标定图片依次读入进行以下操作
for i in range(0,num_images):  
    t = str(i)
    ChessImaR = cv2.imread('left/{}.jpg'.format(t), 0)  # 右视图
    ChessImaL = cv2.imread('right/{}.jpg'.format(t), 0)  # 左视图
    retR, cornersR = cv2.findChessboardCorners(ChessImaR,(6, 9), None)  # 提取右图每一张图片的角点
    retL, cornersL = cv2.findChessboardCorners(ChessImaL,(6, 9), None)  # # 提取左图每一张图片的角点
    if (True == retR) & (True == retL):
        objpoints.append(objp)
        cv2.cornerSubPix(ChessImaR, cornersR, (11, 11), (-1, -1), criteria)  # 亚像素精确化，对粗提取的角点进行精确化
        cv2.cornerSubPix(ChessImaL, cornersL, (11, 11), (-1, -1), criteria)  # 亚像素精确化，对粗提取的角点进行精确化
        imgpointsR.append(cornersR)
        imgpointsL.append(cornersL)

# 相机的单双目标定、及校正
#   右侧相机单独标定
#camera matrix, distortion coefficients, rotation and translation vectors etc.
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints,
                                           imgpointsR,ChessImaR.shape[::-1], 
                                                       None, 
                                                       None)

# Saved Dictory
right_camera_instrincs = {"retR":retR,"mtxR":mtxR,"distR":distR,"rvecsR":rvecsR,
                            "tvecsR":tvecsR}
saved_to_pickle("camera_config/right_camera_intrin",right_camera_instrincs)

#   获取新的相机矩阵后续传递给initUndistortRectifyMap，以用remap生成映射关系
hR, wR = ChessImaR.shape[:2]
OmtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR,(wR, hR), 1, (wR, hR))

#   左侧相机单独标定
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints,imgpointsL,ChessImaL.shape[::-1], None, None)

# Saved Dictory
left_camera_instrincs = {"retL":retL,"mtxL":mtxL,"distL":distL,"rvecsL":rvecsL,
                            "tvecsL":tvecsL}
saved_to_pickle("camera_config/left_camera_intrin",left_camera_instrincs)


#   获取新的相机矩阵后续传递给initUndistortRectifyMap，以用remap生成映射关系
hL, wL = ChessImaL.shape[:2]
OmtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))

# 双目相机的标定
# 设置标志位为cv2.CALIB_FIX_INTRINSIC，这样就会固定输入的cameraMatrix和distCoeffs不变，只求解𝑅,𝑇,𝐸,𝐹
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC

retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(objpoints,imgpointsL,imgpointsR,OmtxL,distL,OmtxR,distR,
                                                           ChessImaR.shape[::-1], criteria_stereo,flags)


# Rectification Parameters
Rectification_parameters = {"retS":retS,"MLS":MLS,"dLS":dLS,"R":R,
                            "T":T,"E":E,"F":F,"MRS":MRS,"dRS":dRS}
saved_to_pickle("camera_config/rectification",Rectification_parameters)

# 利用stereoRectify()计算立体校正的映射矩阵
rectify_scale= 1 # 设置为0的话，对图片进行剪裁，设置为1则保留所有原图像像素
RL, RR, PL, PR, Q, roiL, roiR= cv2.stereoRectify(MLS, dLS, MRS, dRS,
                                                 ChessImaR.shape[::-1], R, T,
                                                 rectify_scale,(0,0))  
# 利用initUndistortRectifyMap函数计算畸变矫正和立体校正的映射变换，实现极线对齐。
Left_Stereo_Map= cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                             ChessImaR.shape[::-1], cv2.CV_16SC2)   

Right_Stereo_Map= cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                              ChessImaR.shape[::-1], cv2.CV_16SC2)

# StereoMap
SavedStereoMap = {"L":Left_Stereo_Map,"R":Right_Stereo_Map}
saved_to_pickle("camera_config/StereoMap",SavedStereoMap)


#立体校正效果显示
for i in range(0,20):  # 以第一对图片为例
    t = str(i)
    frameR = cv2.imread('left/{}.jpg'.format(t), 0)  # 右视图  
    frameL = cv2.imread('right/{}.jpg'.format(t), 0)  # 右视图 
    
    Left_rectified= cv2.remap(frameL,Left_Stereo_Map[0],Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)  # 使用remap函数完成映射
    im_L=Image.fromarray(Left_rectified) # numpy 转 image类
   
    Right_rectified= cv2.remap(frameR,Right_Stereo_Map[0],Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    im_R=Image.fromarray(Right_rectified) # numpy 转 image 类

	#创建一个能同时并排放下两张图片的区域，后把两张图片依次粘贴进去
    width = im_L.size[0]*2
    height = im_L.size[1]

    img_compare = Image.new('RGBA',(width, height))
    img_compare.paste(im_L,box=(0,0))
    img_compare.paste(im_R,box=(640,0))
    
    #在已经极线对齐的图片上均匀画线
    for i in range(1,20):
        len=480/20
        plt.axhline(y=i*len, color='r', linestyle='-')
    plt.imshow(img_compare)
    plt.savefig("outputs/calibrated/{}.png".format(t))
    plt.show()
