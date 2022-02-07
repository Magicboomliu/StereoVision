from utils import unpickle
import json
left_cam = "camera_config/left_camera_intrin"

right_cam  ='camera_config/right_camera_intrin'


rectification_para = 'camera_config/rectification'

left_cam_para = unpickle(left_cam)
right_cam_para = unpickle(right_cam)
rectifications= unpickle(rectification_para)

#camera matrix, distortion coefficients, rotation and translation vectors etc.
rectification = dict()
rectification["R"] = rectifications['R'].tolist()
rectification["T"] = rectifications['T'].tolist()
rectification["E"] = rectifications['E'].tolist()
rectification["F"] = rectifications['F'].tolist()

# 写入data文件
with open('data.json','w') as f:
    json.dump(rectification,f)

