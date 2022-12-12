import os
import cv2
import numpy as np
from glob import glob
import albumentations as A
from tqdm import tqdm

os.makedirs('../dataset/CropScale', exist_ok = True)
os.makedirs('../dataset/CropScale/images', exist_ok = True)
os.makedirs('../dataset/CropScale/labels', exist_ok = True)
class Crop:
    def __init__(self,x_min, y_min, x_max, y_max):
        self.transform = A.Compose([
            A.Crop(x_min, y_min, x_max, y_max)
        ],
        bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

    def __call__(self, im, bboxes, class_labels):
        new = self.transform(image=im, bboxes=bboxes, class_labels=class_labels)  # transformed
        im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])

        return im, labels
class Resize:
    def __init__(self,height, width):
        self.transform = A.Compose([
            A.Resize(height, width, cv2.INTER_CUBIC)
        ],
        bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

    def __call__(self, im, bboxes, class_labels):
        new = self.transform(image=im, bboxes=bboxes, class_labels=class_labels)  # transformed
        im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])

        return im, labels

train_dirs = '../dataset/train'
file_list = glob(train_dirs+'/*.png')
for file_name in tqdm(sorted(file_list)):
    with open(file_name[:-3]+'txt', 'r') as f:
        lines = f.readlines()
    bboxes = [list(map(lambda x: int(x), i[:-1].split(',')))[1:] for i in lines]
    classes = [list(map(lambda x: int(x), i[:-1].split(',')))[0] for i in lines]
    index = 0
    for scale in [1,1.5,2]:
        image = cv2.imread(file_name)
        height, width = image.shape[:2]

        resize_tool = Resize(int(height*scale), int(width*scale))
        try:
            resize_image, resize_label = resize_tool(image, bboxes, classes)
        except:
            print(file_name)
            assert 0
        
        resize_bboxes = [i[1:] for i in resize_label]
        resize_classes = [i[0] for i in resize_label]

        for i in range(0, resize_image.shape[0], 1080):
            for j in range(0, resize_image.shape[1], 1080):
                crop_i_min = min(max(0, resize_image.shape[0]-1281), i)
                crop_j_min = min(max(0, resize_image.shape[1]-1281), j)
                crop_i_max = min(crop_i_min+1280, resize_image.shape[0]-1)
                crop_j_max = min(crop_j_min+1280, resize_image.shape[1]-1)
                crop_tool = Crop(crop_j_min, crop_i_min, crop_j_max, crop_i_max)
                image_crop, label_crop = crop_tool(resize_image, resize_bboxes, resize_classes)
                if len(label_crop) == 0:
                    continue
                cv2.imwrite('../dataset/CropScale/images/'+os.path.basename(file_name)[:-4]+f'_{index}.jpg', image_crop)
                with open('../dataset/CropScale/labels/'+os.path.basename(file_name)[:-4]+f'_{index}.txt', 'w') as w:
                    for l in label_crop:
                        l = [int(label) for label in l]
                        width, height = l[3],l[1]
                        center_x, center_y = (l[1]+l[3]/2)/(crop_j_max-crop_j_min), (l[2]+l[4]/2)/(crop_i_max-crop_i_min)
                        b_width, b_height = l[3]/(crop_j_max-crop_j_min), l[4]/(crop_i_max-crop_i_min)
                        # image_crop = cv2.rectangle(image_crop, (l[1], l[2]), (l[1]+l[3], l[2]+l[4]), [0,0,255], 2)
                        w.write(f'{l[0]} {center_x} {center_y} {b_width} {b_height}\n')
                index+=1