import os
from random import shuffle
import shutil

os.makedirs('../dataset/CropScale/train_set', exist_ok = True)
os.makedirs('../dataset/CropScale/train_set/images', exist_ok = True)
os.makedirs('../dataset/CropScale/train_set/labels', exist_ok = True)
os.makedirs('../dataset/CropScale/val_set', exist_ok = True)
os.makedirs('../dataset/CropScale/val_set/images', exist_ok = True)
os.makedirs('../dataset/CropScale/val_set/labels', exist_ok = True)

files = os.listdir('../dataset/CropScale/images')
shuffle(files)
for f in files[:int(len(files)*0.8)]:
    shutil.copy(f'../dataset/CropScale/images/{f}', f'../dataset/CropScale/train_set/images/{f}')
    shutil.copy(f'../dataset/CropScale/labels/{f[:-3]}txt', f'../dataset/CropScale/train_set/labels/{f[:-3]}txt')
for f in files[int(len(files)*0.8):]:
    shutil.copy(f'../dataset/CropScale/images/{f}', f'../dataset/CropScale/val_set/images/{f}')
    shutil.copy(f'../dataset/CropScale/labels/{f[:-3]}txt', f'../dataset/CropScale/val_set/labels/{f[:-3]}txt')