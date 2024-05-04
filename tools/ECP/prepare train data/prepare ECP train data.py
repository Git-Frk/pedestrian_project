import pandas as pd
import json
import os
import shutil
from tqdm import tqdm
import re
import numpy as np
import random
import warnings

warnings.filterwarnings("ignore")


def prepare_translated_images_for_data_aug(filepath, pedestron_annotation_path, da_image_path, translate_only=False, ratio=1.0):
    """This function prepares a training data using the translated images for training either by augmenting the original data or by creating a dataset with only translated images.
    It sorts all the images to the corresponding city names and returns and saves an annotation file in the format used to train Pedestron
    Note: This function expects to have a folder named train in 'da_image_path' parameter with only the original images(city vise)
    and a pedestron annotation json file in the case were the translated images are used for augmentation.
    - filepath parameter: directory to the translated images
    - pedestron_annotation_path parameter: the annotation file provided by pedestron authors, needs to be in da_image_path folder
    - da_image_path parameter: path of the training dataset to be used for training/fine-tuning Pedestron
    - translate_only [boolean]: If True, a training dataset containing only translated images will be created
    - ratio: the ratio of translated images to be used for augmentation/training (per style)"""

    styles = ['rain', 'winter']

    timage_width = 512
    timage_height = 256

    i = 0  # index to keep track of the item in styles list

    if translate_only:
        # if translate_only the translated images needs to be copied to an empty directory with correct folder and sub-folder names (city names)
        def skip_files(dir, files):
            """this function is for the shutil.copytree() method to instruct it to skip files and only copy directories"""
            return [f for f in files if os.path.isfile(os.path.join(dir, f)) and f != 'pedestron_train.json']

        src = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/augmentation data/leftImg8bit_trainvaltest/train'
        des = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/augmentation data/leftImg8bit_trainvaltest/leftImg8bit/train'
        shutil.copytree(src, des, ignore=skip_files)

    for style in styles:
        print(f'The style being processed: {style}')
        timage_path = os.path.join(filepath, style)
        timage_path = os.path.join(timage_path, style)
        timage_annotations_path = os.path.join(filepath, style)
        timage_annotations_path = os.path.join(timage_annotations_path, 'annotation')

        timages = os.listdir(timage_path)  # translated images
        image_num = int(len(timages) * ratio)  # selecting the ratio of images from translated images
        if image_num == len(timages):
            pass
        else:
            timages = random.sample(timages, image_num)

        tannotations = os.listdir(timage_annotations_path)  # translated image annotations

        with open(pedestron_annotation_path) as jobject:
            pedestron_annotation = json.load(jobject)

        image_df = pd.DataFrame(pedestron_annotation['images'])  # df containing the information of the images
        ann_df = pd.DataFrame(
            pedestron_annotation['annotations'])  # df containing the information of annotations for the images

        if style == 'winter':
            pattern = r'w_leftImg8bit.png$'
        elif style == 'rain':
            pattern = r'r_leftImg8bit.png$'
        elif style == 'night':
            pattern = r'_turbon_leftImg8bit.png$'

        for image in tqdm(timages):
            # print(image)
            img_name = re.sub(pattern, '', image)
            for annotation in tannotations:
                if img_name in annotation:  # when the image name matches the annotation file, that ann file is removed from annotations list. To avoid redundant iterations
                    ann_file = annotation
                    tannotations.remove(annotation)
                    break  # When the annotation file is found there is no need to continue the iteration
            ann_path = os.path.join(timage_annotations_path, ann_file)

            with open(ann_path) as jobject:
                bboxes = json.load(jobject)
            bboxes = bboxes['objects']
            bboxes = [item['bbox'] for item in bboxes]  # bboxes for the image being processed

            # Adding image being processed to the image_df
            filt = (image_df['file_name'].str.contains(img_name)) & (image_df[
                                                                         'license'] == 1)  # getting image information for the corresponding original image(license ==1 condition because in the next iteration(style) there will already be an image with same image name + the condition prefix)
            orgimage = image_df[filt]
            image_id = int(orgimage['id'])
            orgindex = orgimage.index[0]
            image_filepath = orgimage['file_name'].values[0]

            if style == 'winter':
                rpattern = 'w_leftImg8bit.png'
            elif style == 'rain':
                rpattern = 'r_leftImg8bit.png'
            elif style == 'night':
                rpattern = '_turbon_leftImg8bit.png'

            timage_filepath = re.sub(r'_leftImg8bit\.png$', rpattern,
                                     image_filepath)  # getting filepath for the translated image from original filepath

            entry = {'id': image_id + 1,
                     'file_name': timage_filepath,
                     'width': timage_width,
                     'height': timage_height,
                     'date_captured': '2019-07-25 11:20:43.195846',
                     'license': 'translated',
                     'coco_url': '',
                     'flickr_url': ''}

            index = orgindex + 0.5
            image_df.loc[index] = entry
            image_df = image_df.sort_index().reset_index(drop=True)

            image_df['id'][orgindex + 2:] = image_df['id'][orgindex + 2:].apply(lambda x: x + 1)

            # Adding annotations of image being processed to ann_df
            filt = ann_df[
                       'image_id'] == image_id  # getting the annotations for the corresponding original image to get the ann for timage being processed
            org_annotations = ann_df[filt]
            cat_id = org_annotations['category_id']  # information to fill in for the timage annotations
            iscrowd = org_annotations['iscrowd']

            ann_start_index = np.min(
                org_annotations.index)  # getting the first index of the entry with image_id matching the original image
            ann_end_index = np.max(
                org_annotations.index)  # getting the last index of the entry with image_id matching the original image, to insert ann for timage
            t_ann_index = ann_end_index + 1  # to be used later to get the last index of the annotations inserted for the translated image
            id = ann_df.loc[
                ann_end_index, 'id']  # getting id of pedestrian instances, inorder to adjust the ids following timage anns after insertion
            timage_id = image_id + 1  # image index for the translated image

            ann_count = len(bboxes)
            for bbox in bboxes:
                entry = {'id': id + 1,
                         'image_id': timage_id,
                         'category_id': cat_id[ann_start_index],
                         'iscrowd': iscrowd[ann_start_index],
                         'bbox': bbox,
                         'width': timage_width,
                         'height': timage_height}
                index = ann_end_index + 0.5
                ann_df.loc[index] = entry
                ann_df = ann_df.sort_index().reset_index(drop=True)
                ann_df['id'][ann_end_index + 2:] = ann_df['id'][ann_end_index + 2:].apply(lambda x: x + 1)
                id += 1
                ann_end_index += 1
                ann_start_index += 1

            t_ann_index += ann_count  # index of the last entry for the translated image annotation
            ann_df['image_id'][t_ann_index:] = ann_df['image_id'][t_ann_index:].apply(lambda x: x + 1)

            # Copying the translated image being processed to the data augmentation directory
            cities = os.listdir(da_image_path)
            for city in cities:  # Checking what city translated images belongs to
                if city in image:  # When there is a match the loops breaks and the value of city will be the one used
                    break

            translated_image_location = os.path.join(timage_path, image)
            da_img_save_path = os.path.join(da_image_path, city)
            da_img_save_path = os.path.join(da_img_save_path, image)

            shutil.copyfile(translated_image_location, da_img_save_path)

        # Converting the dfs to image dict and annotations dict to modify pedestron annotations file
        if translate_only and i == len(styles)-1:
            # here the dataframes are modified to only have information regarding the translated images
            print('Inside translate_only block')

            original_pedestron_annotation = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/augmentation data/leftImg8bit_trainvaltest/train/pedestron_train.json'

            with open(original_pedestron_annotation) as jobject:
                original_pedestron_ann = json.load(jobject)

            org_image_df = pd.DataFrame(original_pedestron_ann['images'])  # df containing the information of the original/real images only

            columns = image_df.columns
            image_df = image_df.merge(org_image_df, on='file_name', how='left', indicator=True)
            filt = (image_df['_merge'] == 'left_only')
            image_df = image_df.loc[filt, ['id_x', 'file_name', 'width_x', 'height_x', 'date_captured_x', 'license_x', 'coco_url_x', 'flickr_url_x']]
            image_df.columns = columns
            _id = list(range(1, len(image_df['id']) + 1))  # to reset the id from disordered to 1 to len(image_df)
            index = {k: v for k, v in zip(image_df['id'], _id)}  # this dictionary is to hash the original image_id to the modified one, to used with ann_df
            image_df['id'] = _id
            image_df.reset_index(drop=True, inplace=True)

            filt = ann_df['image_id'].isin(np.array(list(index.keys())))
            ann_df = ann_df[filt]
            ann_df.reset_index(drop=True, inplace=True)
            ann_id = list(range(1, len(ann_df['id']) + 1))
            ann_df['id'] = ann_id
            ann_df['image_id'] = ann_df['image_id'].apply(lambda x: index[x])

        modified_image_info = image_df.to_dict(orient='records')
        modified_annotations = ann_df.to_dict(orient='records')

        pedestron_annotation['images'] = modified_image_info
        pedestron_annotation['annotations'] = modified_annotations

        # Saving the modified annotation file
        with open(pedestron_annotation_path, 'w') as jfile:
            json.dump(pedestron_annotation, jfile)

        i += 1

    print('[INFO]: finished processing')


# file_path = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/augmentation data/i2i-turbo'
file_path = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/augmentation data/instruct-pix2pix/translated images'
pedestron_annotations = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/augmentation data/leftImg8bit_trainvaltest/leftImg8bit/train/pedestron_train.json'
augmented_data_path = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/augmentation data/leftImg8bit_trainvaltest/leftImg8bit/train'

prepare_translated_images_for_data_aug(file_path, pedestron_annotations, augmented_data_path, translate_only=True)
