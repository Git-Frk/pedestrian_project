import os
import json
import re
import shutil
import copy
from tqdm import tqdm
import pandas as pd


def select_images_for_data_aug(filtered_image_path, annotation_path, save_path, image_path, style=None, scale=False):
    """This function takes a json file containing the list of images that was filtered using YOLO BC.
    filtered_image_path parameter is the json file containing the list of filtered images
    annotations_path parameter contains all the annotations (of the original unfiltered images)
    save_path parameter is fp for saving the selected images"""

    def _filter_bboxes(coordinates, threshold_area=300):
        """This function is to discard any bounding boxes that  are below a certain area threshold i.e., h*w"""
        x, y, w, h = coordinates
        if w * h >= threshold_area:
            return True
        else:
            return False

    def _scale_coordinates(coord, _scale=0.25):
        """This function scales the bbox coordinates of the original images to fit the translated images
        (x,y,w,h) --> (x',y',w',h')"""
        x, y, w, h = coord

        x1 = int(x * _scale)
        y1 = int(y * _scale)
        x2 = int((x + w) * _scale)
        y2 = int((y + h) * _scale)

        scaled_w = x2 - x1
        scaled_h = y2 - y1

        coord[0], coord[1], coord[2], coord[3], = x1, y1, scaled_w, scaled_h

    # Reading in the JSON file containing filtered image list
    with open(filtered_image_path) as jobject:
        filtered_img_list = json.load(jobject)
        filtered_img_list = filtered_img_list[style]

    annotations = os.listdir(annotation_path)

    file_save_path = os.path.join(save_path, style)
    ann_save_path = os.path.join(file_save_path, 'annotation')
    file_save_path = os.path.join(file_save_path, style)
    os.makedirs(
        file_save_path)  # creating the directory for copying the YOLO filtered images from the cosine filtered images
    os.makedirs(ann_save_path)

    for image in tqdm(filtered_img_list):  # iterating through the filtered images list
        img_name = re.sub(r'_leftImg8bit\.png$', '',
                          image)  # removing all information to get only the image name to get the corresponding annotation file
        for annotation in annotations:
            if img_name in annotation:  # when the image name matches the annotation file, that ann file is removed from annotations list. To avoid redundant iterations
                ann_file = annotation
                annotations.remove(annotation)
                break  # When the annotation file is found there is no need to continue the iteration
        ann_path = os.path.join(annotation_path, ann_file)

        if scale:
            with open(ann_path) as jobject:
                bboxes = json.load(jobject)
                bboxes_objects_copy = copy.deepcopy(bboxes['objects'])

            index = 0
            for box in bboxes['objects']:
                if not _filter_bboxes(box['bbox'],
                                      threshold_area=300):  # This is to remove all bounding boxes which were already small in the original image, when creating new annotations file for the filtered images.
                    bboxes_objects_copy.remove(box)
                    continue
                _scale_coordinates(bboxes_objects_copy[index]['bbox'])
                index += 1
            bboxes['objects'] = bboxes_objects_copy

            # writing the modified annotations file:
            annotation_save_path = os.path.join(ann_save_path, ann_file)
            with open(annotation_save_path, 'w') as jfile:
                json.dump(bboxes, jfile)

        annotation_destination_path = os.path.join(ann_save_path, ann_file)
        shutil.copyfile(ann_path, annotation_destination_path)

        # saving the image for augmentation:
        if style == 'winter':
            img_save_name = img_name + 'w_leftImg8bit.png'
        elif style == 'rain':
            img_save_name = img_name + '_turbor_leftImg8bit.png'
        elif style == 'night':
            img_save_name = img_name + '_turbon_leftImg8bit.png'

        img_save_path = os.path.join(file_save_path, img_save_name)
        img_location = os.path.join(image_path, image)

        shutil.copyfile(img_location, img_save_path)

    # print(bboxes)


yolo_filt_images = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/crops/i2i-turbo crops/night.json'
anns_path = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/cosine images/cosine images(i2i-turbo)/night/annotations'
files_save_path = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/augmentation data'
images_path = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/cosine images/cosine images(i2i-turbo)/night/night'

select_images_for_data_aug(filtered_image_path=yolo_filt_images, annotation_path=anns_path, save_path=files_save_path,
                           image_path=images_path, style='night')


# -------------------------------------------- original code -----------------------------------------------------------
# import os
# import json
# import re
# import shutil
# import copy
# from tqdm import tqdm
# import pandas as pd
#
#
# def select_images_for_data_aug(filtered_image_path, annotation_path, save_path, image_path, style=None, scale=False):
#     """This function takes a json file containing the list of images that was filtered using YOLO BC.
#     filtered_image_path parameter is the json file containing the list of filtered images
#     annotations_path parameter contains all the annotations (of the original unfiltered images)
#     save_path parameter is fp for saving the selected images"""
#
#     def _filter_bboxes(coordinates, threshold_area=300):
#         """This function is to discard any bounding boxes that  are below a certain area threshold i.e., h*w"""
#         x, y, w, h = coordinates
#         if w * h >= threshold_area:
#             return True
#         else:
#             return False
#
#     def _scale_coordinates(coord, _scale=0.25):
#         """This function scales the bbox coordinates of the original images to fit the translated images
#         (x,y,w,h) --> (x',y',w',h')"""
#         x, y, w, h = coord
#
#         x1 = int(x * _scale)
#         y1 = int(y * _scale)
#         x2 = int((x + w) * _scale)
#         y2 = int((y + h) * _scale)
#
#         scaled_w = x2 - x1
#         scaled_h = y2 - y1
#
#         coord[0], coord[1], coord[2], coord[3], = x1, y1, scaled_w, scaled_h
#
#     # Reading in the JSON file containing filtered image list
#     with open(filtered_image_path) as jobject:
#         filtered_img_list = json.load(jobject)
#         filtered_img_list = filtered_img_list[style]
#
#     annotations = os.listdir(annotation_path)
#
#     file_save_path = os.path.join(save_path, style)
#     ann_save_path = os.path.join(file_save_path, 'annotation')
#     file_save_path = os.path.join(file_save_path, style)
#     os.makedirs(
#         file_save_path)  # creating the directory for copying the YOLO filtered images from the cosine filtered images
#     os.makedirs(ann_save_path)
#
#     for image in tqdm(filtered_img_list):  # iterating through the filtered images list
#         img_name = re.sub(r'_leftImg8bit\.png$', '',
#                           image)  # removing all information to get only the image name to get the corresponding annotation file
#         for annotation in annotations:
#             if img_name in annotation:  # when the image name matches the annotation file, that ann file is removed from annotations list. To avoid redundant iterations
#                 ann_file = annotation
#                 annotations.remove(annotation)
#                 break  # When the annotation file is found there is no need to continue the iteration
#         ann_path = os.path.join(annotation_path, ann_file)
#
#         with open(ann_path) as jobject:
#             bboxes = json.load(jobject)
#             bboxes_objects_copy = copy.deepcopy(bboxes['objects'])
#
#         index = 0
#         for box in bboxes['objects']:
#             if not _filter_bboxes(box['bbox'],
#                                   threshold_area=300):  # This is to remove all bounding boxes which were already small in the original image, when creating new annotations file for the filtered images.
#                 bboxes_objects_copy.remove(box)
#                 continue
#             _scale_coordinates(bboxes_objects_copy[index]['bbox'])
#             index += 1
#         bboxes['objects'] = bboxes_objects_copy
#
#         # writing the modified annotations file:
#         annotation_save_path = os.path.join(ann_save_path, ann_file)
#         with open(annotation_save_path, 'w') as jfile:
#             json.dump(bboxes, jfile)
#
#         # saving the image for augmentation:
#         if style == 'winter':
#             img_save_name = img_name + 'w_leftImg8bit.png'
#         elif style == 'rain':
#             img_save_name = img_name + 'r_leftImg8bit.png'
#         elif style == 'night':
#             img_save_name = img_name + 'n_leftImg8bit.png'
#
#         img_save_path = os.path.join(file_save_path, img_save_name)
#         img_location = os.path.join(image_path, image)
#
#         shutil.copyfile(img_location, img_save_path)
#
#     print(bboxes)
#
#
# yolo_filt_images = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/translated images/cosine_images/night/night(50).json'
# anns_path = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/translated images/cosine_images/night/annotations'
# files_save_path = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/augmentation data'
# images_path = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/translated images/cosine_images/night/night'
#
# select_images_for_data_aug(filtered_image_path=yolo_filt_images, annotation_path=anns_path, save_path=files_save_path,
#                            image_path=images_path, style='night')
