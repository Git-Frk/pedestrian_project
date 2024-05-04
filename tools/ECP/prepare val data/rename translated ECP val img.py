import os
import re
import shutil
from tqdm import tqdm


def rename_translated_val_images(validation_source_path, validation_destination_path):
    folder_name = 'translated val'
    validation_destination_path = os.path.join(validation_destination_path, folder_name)
    os.makedirs(validation_destination_path, exist_ok=True)
    styles = ['rain', 'night']

    for style in styles:
        print(f'[INFO]: moving {style} validation images')
        image_path = os.path.join(validation_source_path, style)
        val_images = os.listdir(image_path)
        for image in tqdm(val_images):
            img_name = re.sub(r'_leftImg8bit\.png$', '', image)
            if style == 'winter':
                img_save_name = img_name + 'w_leftImg8bit.png'
            elif style == 'rain':
                img_save_name = img_name + '_turbor_leftImg8bit.png'
            elif style == 'night':
                img_save_name = img_name + '_turbon_leftImg8bit.png'

            img_save_path = os.path.join(validation_destination_path, img_save_name)
            img_location = os.path.join(image_path, image)

            shutil.move(img_location, img_save_path)
    print(f'[INFO] moved images')


val_source = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/augmentation data/leftImg8bit_trainvaltest/val'
val_destination = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/augmentation data/leftImg8bit_trainvaltest/val'

rename_translated_val_images(val_source, val_destination)