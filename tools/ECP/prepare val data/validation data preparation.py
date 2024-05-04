import pandas as pd
import numpy as np
import random
import os
from tqdm import tqdm
import json
import shutil
import re
import warnings

warnings.filterwarnings("ignore")


def prepare_validation_set(org_val_path, translated_val_path, val_annotation_path, translated_only=False, ratio=0.5, copy=False, scale=False, scale_factor=0.25):
    """This function modifies the validation annotation file and image folder to have translated image info and translated images
     - org_val_path parameter: path of the validation dataset(with or without the original images) folder i.e., destination folder
     - translated_val_path parameter: path of the translated validation images
     - val_annotation_path parameter: path of the annotation file for the (original) validation images
     - translated_only [boolean]: If true creates a validation dataset with only translated images
     - ration [0 - 1.0]: determines the ratio of translated image(per style) to be added to the original validation dataset
     - copy [boolean]: whether to copy the translated images to the original validation folder
     - scale [boolean]: scale the bbox, if the translated image dimension is different to the original validation images"""

    def _scale_coordinates(coord):
        """This is a helper function that scales the bbox coordinates of the original images to fit the translated images
        (x,y,w,h) --> (x',y',w',h')"""
        x, y, w, h = coord

        x1 = int(x * scale_factor)
        y1 = int(y * scale_factor)
        x2 = int((x + w) * scale_factor)
        y2 = int((y + h) * scale_factor)

        scaled_w = x2 - x1
        scaled_h = y2 - y1

        return [x1, y1, scaled_w, scaled_h]

    timage_width = 512
    timage_height = 256

    val_images = os.listdir(translated_val_path)  # translated validation images
    image_num = int(len(val_images) * ratio)  # selecting the ratio of images from translated images
    val_set = random.sample(val_images, image_num)

    val_ann_files = ['val_gt_for_mmdetction.json', 'val_gt.json']  # the two validation json file
    val_ann_path = os.path.join(val_annotation_path, val_ann_files[1])
    mmdet_ann_path = os.path.join(val_annotation_path, val_ann_files[0])

    # first modifying gt_val annotation
    with open(val_ann_path) as jobject:
        val_annotations = json.load(jobject)

    val_image_df = pd.DataFrame(val_annotations['images'])
    val_image_df = val_image_df[:-1]
    org_image_df = val_image_df.copy()
    val_ann_df = pd.DataFrame(val_annotations['annotations'])
    org_ann_df = val_ann_df.copy()

    last_img_index = len(val_image_df)
    last_ann_index = len(val_ann_df)

    print('[INFO]: preparing validation dataset')
    for img in tqdm(val_set):
        # print(img)
        if '.png' not in img:
            continue

        if 'w_leftImg8bit.png' in img:
            pattern = r'w_leftImg8bit.png$'
        elif 'r_leftImg8bit.png' in img:
            pattern = r'r_leftImg8bit.png$'
        elif '_turbon_leftImg8bit.png' in img:
            pattern = r'_turbon_leftImg8bit.png$'

        image_name = re.sub(pattern, '', img)
        filt = val_image_df['im_name'].str.contains(image_name)
        org_img = val_image_df[filt]
        org_img_id = org_img['id'].values[0]

        # Inserting items to val_image_df
        img_id = len(val_image_df) + 1

        new_entry = {
            'id': img_id,
            'im_name': img,
            'height': timage_height,
            'width': timage_width
        }
        val_image_df.loc[len(val_image_df)] = new_entry

        # Inserting items to val_ann_df
        last_annotation_id = val_ann_df.iloc[-1, 0]

        filt = (val_ann_df['image_id'] == org_img_id)
        tannotation = val_ann_df[filt]

        tannotation['image_id'] = img_id

        increment_array = np.array(range(1, len(tannotation) + 1))
        tannotation['id'] = last_annotation_id
        tannotation['id'] = (tannotation['id'] + increment_array).apply(int)
        val_ann_df = pd.concat([val_ann_df, tannotation], ignore_index=True)

        # copying the translated images into original validation image folder
        if copy:
            os.makedirs(org_val_path, exist_ok=True)
            timage_path = os.path.join(translated_val_path, img)
            val_image_path = os.path.join(org_val_path, img)

            shutil.copyfile(timage_path, val_image_path)

    # Taking a subset of val_image_df and val_ann_df to only have translated image info
    if translated_only:
        columns = val_image_df.columns
        val_image_df = val_image_df.merge(org_image_df, on='im_name', how='left', indicator=True)
        filt = (val_image_df['_merge'] == 'left_only')
        val_image_df = val_image_df.loc[filt, ['id_x', 'im_name', 'height_x', 'width_x']]
        val_image_df.reset_index(drop=True, inplace=True)
        val_image_df.columns = columns

        columns = val_ann_df.columns
        val_ann_df = val_ann_df.merge(org_ann_df, on='id', how='left', indicator=True)
        filt = (val_ann_df['_merge'] == 'left_only')
        val_ann_df = val_ann_df.loc[
            filt, ['id', 'image_id_x', 'category_id_x', 'iscrowd_x', 'ignore_x', 'bbox_x', 'vis_bbox_x', 'height_x',
                   'vis_ratio_x']]
        val_ann_df.reset_index(drop=True, inplace=True)
        val_ann_df.columns = columns

    if scale:
        val_image_df['height'] = val_image_df['height'].apply(lambda x: int(x / 4))
        val_image_df['width'] = val_image_df['width'].apply(lambda x: int(x / 4))

        val_ann_df['bbox'] = val_ann_df['bbox'].apply(_scale_coordinates)

        height = [bbox[-1] for bbox in val_ann_df['bbox']]
        val_ann_df['height'] = height

    #  modifying val_gt_mmdet annotations
    with open(mmdet_ann_path) as jobject:
        mmdet_annotations = json.load(jobject)

    mmdet_image_df = pd.DataFrame(mmdet_annotations['images'])
    mmdet_image_df = mmdet_image_df[:-1]
    # org_mmdet_image_df = mmdet_image_df.copy()
    mmdet_ann_df = pd.DataFrame(mmdet_annotations['annotations'])
    # org_mmdet_ann_df = mmdet_ann_df.copy()

    image_to_concat = val_image_df[last_img_index:]
    image_to_concat.columns = mmdet_image_df.columns
    annotations_to_concat = val_ann_df[last_ann_index:]
    mmdet_image_df = pd.concat([mmdet_image_df, image_to_concat], ignore_index=True)
    mmdet_ann_df = pd.concat([mmdet_ann_df, annotations_to_concat], ignore_index=True)

    # Taking a subset of mmdet_image_df and mmdet_ann_df to only have translated image info
    if translated_only and np.all(org_image_df.values == mmdet_image_df.values) and np.all(
            org_ann_df.values == mmdet_ann_df.values):
        columns = mmdet_image_df.columns  # The subset of val_image_df has already been taken, and it is same as mmdet_image_df except for column name
        mmdet_image_df = val_image_df.copy()
        mmdet_image_df.columns = columns

        columns = mmdet_ann_df.columns
        mmdet_ann_df = val_ann_df.copy()
        mmdet_ann_df.columns = columns

    # Converting the dataframes to dictionary
    modified_image_info = val_image_df.to_dict(orient='records')
    modified_annotations = val_ann_df.to_dict(orient='records')

    val_annotations['images'] = modified_image_info
    val_annotations['annotations'] = modified_annotations

    mmdet_modified_image_info = mmdet_image_df.to_dict(orient='records')
    mmdet_modified_annotation = mmdet_ann_df.to_dict(orient='records')

    mmdet_annotations['images'] = mmdet_modified_image_info
    mmdet_annotations['annotations'] = mmdet_modified_annotation

    # Writing the modified val_gt info in the json file
    save_path = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/augmentation data/leftImg8bit_trainvaltest/leftImg8bit/val_all_in_folder'
    os.makedirs(save_path, exist_ok=True)

    val_gt_save_path = os.path.join(save_path, val_ann_files[1])
    with open(val_gt_save_path, 'w') as jfile:
        json.dump(val_annotations, jfile)

    # Writing the modified val_gt_mmdet info in the json file
    mmdet_save_path = os.path.join(save_path, val_ann_files[0])
    with open(mmdet_save_path, 'w') as jfile:
        json.dump(mmdet_annotations, jfile)

    print('[INFO]: finished processing')


val_org = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/augmentation data/leftImg8bit_trainvaltest/leftImg8bit/val_all_in_folder'
val_trans = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/augmentation data/leftImg8bit_trainvaltest/val/translated val'
val_ann = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data preparation- DA/validation data prep/CityPersons'

prepare_validation_set(val_org, val_trans, val_ann, translated_only=True, copy=True)
