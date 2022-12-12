import os
import cv2
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("result",
                    help="Detected result")

args = parser.parse_args()
path = args.result
color = [[255,0,0], [0,0,255], [0,255,0],[125,62,125]]
files = os.listdir(f'{path}/labels')
with open('ans.csv', 'w') as w:
    for f in tqdm(sorted(files)):
        img = cv2.imread(os.path.join('test', f'{f[:-3]}png'))
        height, width = img.shape[:2]
        with open(f'{path}/labels/{f}', 'r') as d:
            for det in d.readlines():
                label, b_x_center, b_y_center, b_width, b_height = det.split()
                b_w, b_h = float(b_width) * width, float(b_height) * height
                if b_w == 0 or b_h == 0:
                    print(label, b_x_center, b_y_center, b_width, b_height)
                    continue
                b_x_center, b_y_center = float(b_x_center) * width, float(b_y_center) * height
                x_min = b_x_center - b_w/2
                y_min = b_y_center - b_h/2
                w.write(','.join([f[:-4], label, str(int(x_min)), str(int(y_min)), str(int(b_w)), str(int(b_h))]) + '\n')
        #         x_max = b_x_center + b_w/2
        #         y_max = b_y_center + b_h/2
        #         cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color[int(label)], 2)
        # cv2.imshow(f, cv2.resize(img, (1920,960)))
        # cv2.waitKey()
        # cv2.destroyAllWindows()        