
import pickle
import cv2
def saved_to_pickle(file_name,dct):
    with open(file_name,'wb') as f1:
        pickle.dump(dct,f1)



def unpickle(file):
    with open(file, 'rb') as fo:
        dct = pickle.load(fo, encoding='bytes')
    return dct

def preprocssing(img_L,img_R):
    dicts = unpickle("camera_config/StereoMap")
    Left_Stereo_Map = dicts["L"]
    Right_Stereo_Map = dicts["R"]
    Left_rectified= cv2.remap(img_R,Left_Stereo_Map[0],Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)  # 使用remap函数完成映射
    Right_rectified= cv2.remap(img_L,Right_Stereo_Map[0],Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    right_view = Right_rectified
    left_view = Left_rectified

    return right_view,left_view



if __name__=="__main__":
    saved_pickle_file = "camera_config/right_camera_intrin"
    dicts = unpickle(saved_pickle_file)
    print(dicts)
