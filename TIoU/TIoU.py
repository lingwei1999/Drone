import numpy as np
import math
import os
import cv2
from tqdm import tqdm
import pandas as pd
import openpyxl
def find_interArea(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    return [xmin, ymin, xmax, ymax]

def cal_iou(box1, box2):
    """
    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    :return: 
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    # 计算每个矩形的面积
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # b1的面积
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # b2的面积
 
    # 计算相交矩形
    xmin, ymin, xmax, ymax = find_interArea(box1, box2)
 
    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    a1 = w * h  # C∩G的面积
    a2 = s1 + s2 - a1
    iou = a1 / a2 #iou = a1/ (s1 + s2 - a1)
    return [iou, a1]

classes = ['car', 'hov', 'people', 'motor']
test_path = 'predict'
test_model = os.listdir(f'./{test_path}')
# test_model = ['yolov7x-1920_origin-conf0.25', 'yolov7x-1920_origin-conf0.419', 
#                 'yolov7x-CropMS_2560-conf0.25', 'yolov7x-CropMS_2560-conf0.449', 
#                 'yolov7x-CropMS_ft_woHyper_2560-conf0.25', 'yolov7x-CropMS_ft_woHyper_2560-conf0.49', 
#                 'yolov7x-CropMS_ft_withHyper_2560-conf0.25', 'yolov7x-CropMS_ft_withHyper_2560-conf0.443']
c_recall = {'model': test_model}
c_precision = {'model': test_model}
c_score = {'model': test_model}
Final_score = {'model': test_model}
for i in classes:
    c_recall[i] = []
    c_precision[i] = []
    c_score[i] = []
    Final_score[i] = []

c_recall['all'] = []
c_precision['all'] = []
c_score['all'] = []
Final_score['all'] = []

for model in test_model:
    print('================================================')
    print(model)
    files = os.listdir('../dataset/val_set/images')
    recall = [[],[],[],[]]
    precision = [[],[],[],[]]
    score_dis = [[],[],[],[]]
    for f_name in tqdm(files):
        img = cv2.imread(f'../dataset/val_set/images/{f_name[:-4]}.jpg')
        Height, Width = img.shape[:2]

        gt = [[],[],[],[]]
        with open(f'../dataset/val_set/labels/{f_name[:-4]}.txt', 'r') as f:
            for i in f.readlines():
                line = list(map(lambda x: float(x), i.split()))
                x_center, y_center = line[1]*Width, line[2]*Height
                width, height = line[3]*Width, line[4]*Height
                gt[int(line[0])].append([x_center-width/2, y_center-height/2, x_center+width/2, y_center+height/2])
        pred = [[],[],[],[]]
        with open(f'./{test_path}/{model}/labels/{f_name[:-4]}.txt', 'r') as f:
            for i in f.readlines():
                line = list(map(lambda x: float(x), i.split()))
                x_center, y_center = line[1]*Width, line[2]*Height
                width, height = line[3]*Width, line[4]*Height
                pred[int(line[0])].append([x_center-width/2, y_center-height/2, x_center+width/2, y_center+height/2])
                
        for c in range(len(gt)):
            for g in gt[c]:
                if len(pred[c]) == 0:
                    recall[c].append(0)
                    continue
                IoU, interArea= max([cal_iou(g, p) for p in pred[c]], key = lambda x: x[0])
                if IoU>=0.5:
                    gt_area = (g[2]-g[0]) * (g[3]-g[1])
                    recall[c].append(IoU*(interArea/gt_area))
                else:
                    recall[c].append(0)
        for c in range(len(pred)):
            for p in pred[c]:
                if len(gt[c]) == 0:
                    precision[c].append(0)
                    continue
                idx, IoU, interArea = max([[idx] + cal_iou(g, p) for idx, g in enumerate(gt[c])], key = lambda x: x[1])
                if IoU>=0.5:
                    pred_area = (p[2]-p[0]) * (p[3]-p[1])
                    inter_gp = find_interArea(gt[c][idx], p)
                    precision[c].append(IoU * (1 - (sum([cal_iou(p, g)[1] - cal_iou(inter_gp, g)[1] for gid, g in enumerate(gt[c]) if gid != idx]))/pred_area))
                else:
                    precision[c].append(0)        
        for c in range(len(gt)):
            for g in gt[c]:
                if len(pred[c]) == 0:
                    score_dis[c].append(0)
                    continue
                idx, IoU, interArea = max([[idx] + cal_iou(g, p) for idx, p in enumerate(pred[c])], key = lambda x: x[1])
                if IoU>=0.5:
                    gt_center = np.array([(g[2]+g[0])/2, (g[3]+g[1])/2], dtype = 'float32')
                    pred_center = np.array([(pred[c][idx][2] + pred[c][idx][0])/2, (pred[c][idx][3] + pred[c][idx][1])/2], dtype = 'float32')
                    dist = (np.linalg.norm([gt_center - pred_center])**2)/25

                    score_dis[c].append(math.exp(-dist))
                else:
                    score_dis[c].append(0)

    for idx, i in enumerate(classes):
        c_recall[i].append(sum(recall[idx])/len(recall[idx]))
        c_precision[i].append(sum(precision[idx])/len(precision[idx]))
        c_score[i].append(sum(score_dis[idx])/len(score_dis[idx]))
        Final_score[i].append(((c_recall[i][-1] * c_precision[i][-1] * c_score[i][-1]) * 3) / (c_recall[i][-1] * c_precision[i][-1] + c_precision[i][-1] * c_score[i][-1] + c_recall[i][-1] * c_score[i][-1]))
    c_recall['all'].append(sum([sum(r) for r in recall])/sum([len(r) for r in recall]))
    c_precision['all'].append(sum([sum(p) for p in precision])/sum([len(p) for p in precision]))
    c_score['all'].append(sum([sum(s) for s in score_dis])/sum([len(s) for s in score_dis]))
    Final_score['all'].append(((c_recall['all'][-1] * c_precision['all'][-1] * c_score['all'][-1]) * 3) / (c_recall['all'][-1] * c_precision['all'][-1] + c_precision['all'][-1] * c_score['all'][-1] + c_recall['all'][-1] * c_score['all'][-1]))

writer = pd.ExcelWriter(f'{test_path}.xlsx', engine='openpyxl')

result_r = pd.DataFrame(c_recall)
result_r.to_excel(writer, sheet_name='Recall')
result_p = pd.DataFrame(c_precision)
result_p.to_excel(writer, sheet_name='Precision')
result_s = pd.DataFrame(c_score)
result_s.to_excel(writer, sheet_name='Score')
result = pd.DataFrame(Final_score)
result.to_excel(writer, sheet_name='Final Score')
writer.save()