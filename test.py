import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
import tqdm
from model.locator import Crowd_locator
from misc.utils import *
from PIL import Image, ImageOps
import  cv2
import json

dataset = 'NWPU'
dataRoot = '../ProcessedData/' + dataset
test_list = 'val.txt'

dataset = 'part_B_final'
dataRoot = '../ProcessedData/' + dataset
test_list = 'val.txt'

GPU_ID = '0,1'
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
torch.backends.cudnn.benchmark = True

netName = 'HR_Net'
# model_path = '../PretrainedCrowdLocModel/NWPU-HR-ep_241_F1_0.802_Pre_0.841_Rec_0.766_mae_55.6_mse_330.9.pth'
model_path ="/home/lishuang/Disk/gitlab/traincode/crowd_counting/PyTorch_Pretrained/NWPU-HR-ep_241_F1_0.802_Pre_0.841_Rec_0.766_mae_55.6_mse_330.9.pth"
# netName = 'VGG16_FPN'
# model_path = '../PretrainedCrowdLocModel/NWPU-VGG-ep_361_F1_0.770_Pre_0.802_Rec_0.741_mae_62.7_mse_299.2.pth'

out_file_name= './saved_exp_results/' + dataset + '_' + netName + '_' + test_list




if dataset == 'NWPU':
    mean_std = ([0.446139603853, 0.409515678883, 0.395083993673], [0.288205742836, 0.278144598007, 0.283502370119])

if dataset == 'part_B_final':
    mean_std = ([0.446139603853, 0.409515678883, 0.395083993673], [0.288205742836, 0.278144598007, 0.283502370119])
    

img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
restore = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

def main():

    txtpath = os.path.join(dataRoot, test_list)
    with open(txtpath) as f:
        lines = f.readlines()                            
    test(lines, model_path)


def get_boxInfo_from_Binar_map(Binar_numpy, min_area=3):
    Binar_numpy = Binar_numpy.squeeze().astype(np.uint8)
    assert Binar_numpy.ndim == 2
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(Binar_numpy, connectivity=4)  # centriod (w,h)

    boxes = stats[1:, :]
    points = centroids[1:, :]
    index = (boxes[:, 4] >= min_area)
    boxes = boxes[index]
    points = points[index]
    pre_data = {'num': len(points), 'points': points}
    return pre_data, boxes

def read_box_gt(box_gt_file):
    gt_data = {}
    with open(box_gt_file) as f:
        for line in f.readlines():
            line = line.strip().split(' ')

            line_data = [int(i) for i in line]
            idx, num = [line_data[0], line_data[1]]
            points_r = []
            if num > 0:
                points_r = np.array(line_data[2:]).reshape(((len(line) - 2) // 5, 5))
                gt_data[idx] = {'num': num, 'points': points_r[:, 0:2], 'sigma': points_r[:, 2:4], 'level': points_r[:, 4]}
            else:
                gt_data[idx] = {'num': 0, 'points': [], 'sigma': [], 'level': []}

    return gt_data

def read_box_gt_json(json_name):
    with open(json_name) as f:
        ImgInfo = json.load(f)
    human_num = ImgInfo["human_num"]
    ann_points=np.array(ImgInfo['points'])
    false_sigma=np.ones_like(ann_points)*30
    false_level=np.ones([ann_points.shape[0],])*3
    if human_num > 0:
        return {'num': human_num, 'points': ann_points, 'sigma': false_sigma,
                        'level': false_level}
    else:
        return {'num': 0, 'points': [], 'sigma': [], 'level': []}


def test(file_list, model_path):

    net = Crowd_locator(netName,GPU_ID,pretrained=True)
    net.cuda()
    net.load_state_dict(torch.load(model_path))
    net.eval()

    gts = []
    preds = []
    cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter(), 'nae': AverageMeter()}

    if dataset == 'NWPU':
        box_gt_txt='val_gt_loc.txt'
        box_gt_Info = read_box_gt(os.path.join(dataRoot, box_gt_txt))

    file_list = tqdm.tqdm(file_list)
    for infos in file_list:
        filename = infos.split()[0]

        # imgname = os.path.join(dataRoot, 'test', filename + '.jpg')
        imgname = os.path.join(dataRoot, 'images', filename + '.jpg')
        if dataset == 'NWPU':
            gt_data = box_gt_Info[int(filename)]
        if dataset == 'part_B_final':
            imgname=os.path.join(dataRoot, filename + '.jpg')
            json_name = os.path.join(dataRoot, filename.replace("images/", "jsons/") + '.json')
            gt_data=read_box_gt_json(json_name)
        img = Image.open(imgname)

        if img.mode == 'L':
            img = img.convert('RGB')
        img = img_transform(img)[None, :, :, :]
        slice_h, slice_w = 512,1024
        slice_h, slice_w = slice_h, slice_w
        with torch.no_grad():
            img = Variable(img).cuda()
            b, c, h, w = img.shape
            crop_imgs, crop_dots, crop_masks = [], [], []
            if h * w < slice_h * 2 * slice_w * 2 and h % 16 == 0 and w % 16 == 0:
                [pred_threshold, pred_map, __] = [i.cpu() for i in net(img, mask_gt=None, mode='val')]
            else:
                if h % 16 != 0:
                    pad_dims = (0, 0, 0, 16 - h % 16)
                    h = (h // 16 + 1) * 16
                    img = F.pad(img, pad_dims, "constant")


                if w % 16 != 0:
                    pad_dims = (0, 16 - w % 16, 0, 0)
                    w = (w // 16 + 1) * 16
                    img = F.pad(img, pad_dims, "constant")


                for i in range(0, h, slice_h):
                    h_start, h_end = max(min(h - slice_h, i), 0), min(h, i + slice_h)
                    for j in range(0, w, slice_w):
                        w_start, w_end = max(min(w - slice_w, j), 0), min(w, j + slice_w)
                        crop_imgs.append(img[:, :, h_start:h_end, w_start:w_end])
                        mask = torch.zeros(1,1,img.size(2), img.size(3)).cpu()
                        mask[:, :, h_start:h_end, w_start:w_end].fill_(1.0)
                        crop_masks.append(mask)
                crop_imgs, crop_masks =  torch.cat(crop_imgs, dim=0), torch.cat(crop_masks, dim=0)

                # forward may need repeatng
                crop_preds, crop_thresholds = [], []
                nz, period = crop_imgs.size(0), 4
                for i in range(0, nz, period):
                    [crop_threshold, crop_pred, __] = [i.cpu() for i in net(crop_imgs[i:min(nz, i+period)],mask_gt = None, mode='val')]
                    crop_preds.append(crop_pred)
                    crop_thresholds.append(crop_threshold)

                crop_preds = torch.cat(crop_preds, dim=0)
                crop_thresholds = torch.cat(crop_thresholds, dim=0)

                # splice them to the original size
                idx = 0
                pred_map = torch.zeros(b, 1, h, w).cpu()
                pred_threshold = torch.zeros(b, 1, h, w).cpu().float()
                for i in range(0, h, slice_h):
                    h_start, h_end = max(min(h - slice_h, i), 0), min(h, i + slice_h)
                    for j in range(0, w, slice_w):
                        w_start, w_end = max(min(w - slice_w, j), 0), min(w, j + slice_w)
                        pred_map[:, :, h_start:h_end, w_start:w_end] += crop_preds[idx]
                        pred_threshold[:, :, h_start:h_end, w_start:w_end] += crop_thresholds[idx]
                        idx += 1
                mask = crop_masks.sum(dim=0)
                pred_map = (pred_map / mask)
                pred_threshold = (pred_threshold / mask)

            a = torch.ones_like(pred_map)
            b = torch.zeros_like(pred_map)
            binar_map = torch.where(pred_map >= pred_threshold, a, b)

            imgdst = cv2.imread(imgname)  # bgr
            pred_data, boxes = get_boxInfo_from_Binar_map(binar_map.cpu().numpy())
            pred_p=pred_data['points']
            point_r_value = 5
            for i in range(pred_p.shape[0]):
                cv2.circle(imgdst, (int(pred_p[i][0]), int(pred_p[i][1])), point_r_value, (0, 255, 0), -1)  # tp: green
            cv2.imwrite('./saved_exp_results/img_save/'  + os.path.basename(filename) +'_'+str(pred_data['num'])+ '.jpg',imgdst)


            gt_count, pred_cnt = gt_data['num']*1.0, pred_data['num']
            s_mae = abs(gt_count - pred_cnt)
            s_mse = ((gt_count - pred_cnt) * (gt_count - pred_cnt))
            cnt_errors['mae'].update(s_mae)
            cnt_errors['mse'].update(s_mse)
            if gt_count != 0:
                s_nae = (abs(gt_count - pred_cnt) / gt_count)
                cnt_errors['nae'].update(s_nae)

            mae = cnt_errors['mae'].avg
            mse = np.sqrt(cnt_errors['mse'].avg)
            nae = cnt_errors['nae'].avg
            print("mae=",mae,"mse=",mse,"nae=",nae)

            with open(out_file_name, 'a') as f:

                f.write(filename + ' ')
                f.write(str(pred_data['num']) + ' ')
                for ind,point in enumerate(pred_data['points'],1):
                    if ind < pred_data['num']:
                        f.write(str(int(point[0])) + ' ' + str(int(point[1])) + ' ')
                    else:
                            f.write(str(int(point[0])) + ' ' + str(int(point[1])))
                f.write('\n')
                f.close()

        # record.close()

if __name__ == '__main__':
    main()




