
import cv2
from utils import preprocssing
import matplotlib.pyplot as plt
from algorithms import opencv_bm, opencv_sgm


if __name__=="__main__":

    # left_view = cv2.imread("test/MiddleBurry/2003/cones/im2.png",0)
    # right_view = cv2.imread("test/MiddleBurry/2003/cones/im6.png",0)
    
    left_image = cv2.imread('left/10.jpg')  # 右视图  
    right_image = cv2.imread('right/10.jpg')  # 右视图

    

    # # left_image = cv2.imread('test/left/31.jpg',0)  # 右视图  
    # # right_image = cv2.imread('test/right/31.jpg',0)  # 右视图

    left_view,right_view = preprocssing(left_image,right_image)

    # # cv2.imwrite("left_view.png",left_view)
    # # cv2.imwrite("right_image.png",right_view)

    
    # # from PIL import Image
    # # im_R=Image.fromarray(right_view) # numpy 转 image 类
    # # im_L = Image.fromarray(left_view)

	# # #创建一个能同时并排放下两张图片的区域，后把两张图片依次粘贴进去
    # # width = im_L.size[0]*2
    # # height = im_L.size[1]

    # # img_compare = Image.new('RGBA',(width, height))
    # # img_compare.paste(im_L,box=(0,0))
    # # img_compare.paste(im_R,box=(640,0))
    
    # # #在已经极线对齐的图片上均匀画线
    # # for i in range(1,20):
    # #     len=480/20
    # #     plt.axhline(y=i*len, color='r', linestyle='-')
    # # plt.imshow(img_compare)
    # # plt.show()

    # window_size =25
    # #disp = opencv_sgm(left_view,right_view,window_size=window_size,max_disp=64,SAD_block_size=3,vis=True)
    # disp = opencv_bm(left_data=left_view,right_data=right_view,numDisparities=64,blockSize=window_size)


    # plt.figure(figsize=(10,6))
    # plt.subplot(1,3,1)
    # plt.axis("off")
    # plt.imshow(left_view,cmap='gray')
    # plt.subplot(1,3,2)
    # plt.axis("off")
    # plt.imshow(right_view,cmap='gray')
    # plt.subplot(1,3,3)
    # plt.axis("off")
    # plt.imshow(disp,cmap='gray')
    # plt.savefig("outputs/bm_{}".format(window_size))
    # plt.show()

    plt.figure(figsize=(10,6))
    plt.subplot(2,2,1)
    plt.axis("off")
    plt.imshow(left_image,cmap='gray')
    plt.subplot(2,2,2)
    plt.axis("off")
    plt.imshow(right_image,cmap='gray')
    plt.subplot(2,2,3)
    plt.axis("off")
    plt.imshow(left_view,cmap='gray')
    plt.subplot(2,2,4)
    plt.axis("off")
    plt.imshow(right_view,cmap='gray')
    # plt.savefig("outputs/bm_{}".format(window_size))
    plt.show()
    