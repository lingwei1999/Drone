import os
import shutil
import numpy as np
from PIL import Image

from src.noise import *
from src.flip import *
from src.sharpen import *
from src.blur import *
from src.crop import *
from src.cutout import *
from src.lightness import *
from src.contrast import *
from src.deform import *
from src.distortion import *
from src.vignetting import *
from tqdm import tqdm
if __name__ == "__main__":
    img_path = '../dataset/train_set/images'
    label_path = '../dataset/train_set/labels'
    files = os.listdir(img_path)
    
    os.makedirs('../dataset/Augment', exist_ok=True)
    os.makedirs('../dataset/Augment/images', exist_ok=True)
    os.makedirs('../dataset/Augment/labels', exist_ok=True)

    for f_path in tqdm(sorted(files)):

        (name, appidx) = os.path.splitext(f_path)
        img = np.array(Image.open(os.path.join(img_path, f_path))) 
        Image.fromarray(img).save(f'../dataset/Augment/images/{name}.jpg')
        shutil.copyfile(f'{label_path}/{name}.txt', f'../dataset/Augment/labels/{name}.txt')

        '''
        需要重新标注
        Need to relabled
        '''
        # # crop
        # crop_img = crop(np.copy(img))
        # crop_img.save(name + "_crop" + appidx)

        # # deform
        # deform_img = deform(np.copy(img))
        # deform_img.save(name + "_deform" + appidx)

        # # distortion
        # distortion_img = distortion(np.copy(img))
        # distortion_img.save(name + "_distortion" + appidx)


        '''
        自动计算变换后的标签位置
        Automatically calculate the label position after the transformation
        '''
        # noise
        noise_img = addNoise(np.copy(img))
        noise_img.save(f"../dataset/Augment/images/{name}_noise.jpg")
        shutil.copyfile(f'{label_path}/{name}.txt', f'../dataset/Augment/labels/{name}_noise.txt')

        # flip
        flip_img = flip(np.copy(img))
        flip_img.save(f"../dataset/Augment/images/{name}_flip.jpg")
        with open(f'../dataset/Augment/labels/{name}_flip.txt', "w") as outfile:
            with open(f'{label_path}/{name}.txt', "r") as infile:
                for line in infile.readlines():
                    words = line.split(" ")
                    horizontal_coord = float(words[1])
                    outfile.write(words[0] + " " + str(format(1-horizontal_coord, ".6f")) + " " + words[2] + " " + words[3] + " " + words[4]+"\n")
        # saveFlipLabel(name)

        # sharpen
        sharpen_img = sharpen(np.copy(img))
        sharpen_img.save(f"../dataset/Augment/images/{name}_sharp.jpg")
        shutil.copyfile(f'{label_path}/{name}.txt', f'../dataset/Augment/labels/{name}_sharp.txt')
        # saveSharpenLabel(name)

        # blur
        blur_img = blur(np.copy(img))
        blur_img.save(f"../dataset/Augment/images/{name}_blur.jpg")
        shutil.copyfile(f'{label_path}/{name}.txt', f'../dataset/Augment/labels/{name}_blur.txt')
        # saveBlurLabel(name)

        # # cutout
        # cutout_img = cutout(np.copy(img))
        # cutout_img.save(f"../dataset/Augment/images/{name}_cutout.png")
        # shutil.copyfile(f'{label_path}/{name}.txt', f'../dataset/Augment/labels/{name}_cutout.txt')
        # # saveCutoutLabel(name)

        # lightness
        ## brightness
        brightness_img = brightness(np.copy(img))
        brightness_img.save(f"../dataset/Augment/images/{name}_brightness.jpg")
        shutil.copyfile(f'{label_path}/{name}.txt', f'../dataset/Augment/labels/{name}_brightness.txt')
        # saveBrightnessLabel(name)

        ## darkness
        darkness_img = darkness(np.copy(img))
        darkness_img.save(f"../dataset/Augment/images/{name}_darkness.jpg")
        shutil.copyfile(f'{label_path}/{name}.txt', f'../dataset/Augment/labels/{name}_darkness.txt')
        # saveDarknessLabel(name)

        # contrast
        contrast_img = contrast(np.copy(img))
        contrast_img.save(f"../dataset/Augment/images/{name}_contrast.jpg")
        shutil.copyfile(f'{label_path}/{name}.txt', f'../dataset/Augment/labels/{name}_contrast.txt')
        # saveContrastLabel(name)

        # vignetting
        vignetting_img = vignetting(np.copy(img))
        vignetting_img.save(f"../dataset/Augment/images/{name}_vignetting.jpg")
        shutil.copyfile(f'{label_path}/{name}.txt', f'../dataset/Augment/labels/{name}_vignetting.txt')
        # saveVignettingLabel(name)
