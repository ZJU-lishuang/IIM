from PIL import Image
from scipy import io as sio
import os
import json
import cv2

SHOW_DOT=False

data_path="/home/lishuang/Disk/download/part_B_final/test_data"  #train_data  test_data
img_path = data_path + '/images'
gt_path = data_path + '/ground_truth'
json_path=data_path + '/jsons'

if not os.path.exists(json_path):
    os.mkdir(json_path)
if SHOW_DOT:
    show_path = data_path + "/show_dot"
    if not os.path.exists(show_path):
        os.mkdir(show_path)
data_files = [filename for filename in os.listdir(img_path) \
                if os.path.isfile(os.path.join(img_path,filename))]
num_samples = len(data_files)
for index in range(num_samples):
    json_dict = {}  # ls
    fname = data_files[index]
    img = Image.open(os.path.join(img_path ,fname))
    if img.mode == 'L':
        img = img.convert('RGB')

    den = sio.loadmat(os.path.join(gt_path,'GT_'+os.path.splitext(fname)[0] + '.mat'))
    annpoints=den['image_info'].item().item()[0]
    person_num=den['image_info'].item().item()[1]
    half_box_w=12
    half_box_h=12
    annbox=[]
    for index in range(len(annpoints)):
        center_x,center_y=annpoints[index]
        x_start=center_x-half_box_w
        x_end=center_x+half_box_w
        y_start=center_y-half_box_h
        y_end=center_y+half_box_h
        annbox.append([x_start,y_start,x_end,y_end])

    if SHOW_DOT:
        imgdst = cv2.imread(os.path.join(img_path ,fname))  # bgr
        pred_p=annpoints
        point_r_value = 5
        for i in range(pred_p.shape[0]):
            cv2.circle(imgdst, (int(pred_p[i][0]), int(pred_p[i][1])), point_r_value, (0, 255, 0), -1)  # tp: green
        cv2.imwrite(os.path.join(show_path ,fname),imgdst)

    json_dict["img_id"]=fname
    json_dict["human_num"]=person_num.item()
    json_dict["points"]=annpoints.tolist()
    json_dict["boxes"]=annbox
    json_file=os.path.join(json_path ,os.path.splitext(fname)[0]+".json")
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()