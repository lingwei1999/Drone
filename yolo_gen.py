from glob import glob
import random
import os
import cv2
from tqdm import tqdm

datasets = './dataset/train'
dst = './dataset/train_set'
os.makedirs(dst, exist_ok=True)
os.makedirs(os.path.join(dst, 'images'), exist_ok=True)
os.makedirs(os.path.join(dst, 'labels'), exist_ok=True)

color = [[255,0,0], [0,0,255], [0,255,0],[0,0,0]]
img_files = glob(os.path.join(datasets, '*.png'))
random.shuffle(img_files)
for img_name in tqdm(img_files[:int(len(img_files)*0.8)]):
    img = cv2.imread(img_name)

    cv2.imwrite(os.path.join(dst, 'images', os.path.basename(img_name)[:-3]+'jpg'), img)
    with open(os.path.join(dst, 'labels', os.path.basename(img_name)[:-3]+'txt'), 'w') as w:
        img_width = 1.0/img.shape[1]
        img_height = 1.0/img.shape[0]
        with open(img_name[:-3]+'txt', 'r') as r:
            for line in r.readlines():
                label, xmin, ymin, width, height = [int(i) for i in line.split(',')]
                x_center = (xmin+width/2)*img_width
                y_center = (ymin+height/2)*img_height
                b_width = width*img_width
                b_height = height*img_height
                w.write(' '.join([str(label), str(x_center), str(y_center), str(b_width), str(b_height), '\n']))
        #         x_min = xmin
        #         y_min = ymin
        #         x_max = xmin + width
        #         y_max = ymin + height
        #         cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color[int(label)], 2)
        # cv2.imshow(img_name, img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

dst = 'val_set'
os.makedirs(dst, exist_ok=True)
os.makedirs(os.path.join(dst, 'images'), exist_ok=True)
os.makedirs(os.path.join(dst, 'labels'), exist_ok=True)

for img_name in tqdm(img_files[int(len(img_files)*0.8):]):
    img = cv2.imread(img_name)

    cv2.imwrite(os.path.join(dst, 'images', os.path.basename(img_name)[:-3]+'jpg'), img)
    with open(os.path.join(dst, 'labels', os.path.basename(img_name)[:-3]+'txt'), 'w') as w:
        img_width = 1.0/img.shape[1]
        img_height = 1.0/img.shape[0]
        with open(img_name[:-3]+'txt', 'r') as r:
            for line in r.readlines():
                label, xmin, ymin, width, height = [int(i) for i in line.split(',')]
                x_center = (xmin+width/2)*img_width
                y_center = (ymin+height/2)*img_height
                b_width = width*img_width
                b_height = height*img_height
                w.write(' '.join([str(label), str(x_center), str(y_center), str(b_width), str(b_height), '\n']))
