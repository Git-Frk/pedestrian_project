from PIL import Image
from tqdm import tqdm
import json
import numpy as np
import cv2
import os
from image_info import ImagesInfo
import random

ann_path = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/original images/annotations'

original_path = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/original images/ images'
translated_path = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/cosine images/cosine images(i2i-turbo)/night/night'

ped_save_path = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/crops/ped_crop'
random_save_path = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/crops/random_crop'
translated_crop_save_path = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/crops/i2i-turbo crops/night'


class Crop(ImagesInfo):
    def __init__(self):
        super().__init__()
        self.__original_image_path = None
        self.__translated_image_path = None
        self.__original_save_path = None
        self.__translated_save_path = None
        self.__original_dimension = None
        self.__translated_dimension = None

    @staticmethod
    def __compute_area(coords):
        area = 0
        for coord in coords:
            w, h = coord[-2], coord[-1]
            _area = w * h
            area += _area
        return area

    @staticmethod
    def is_intersect(random_bbox, image_bboxes):
        """This function checks if the randomly cropped patch
         intersects with any of the patches containing pedestrians in that image"""
        rx, ry, rw, rh = random_bbox

        for bbox in image_bboxes:
            x, y, w, h = bbox

            rbbox_left_x = rx
            rbbox_right_x = rx + rw
            rbbox_top_y = ry
            rbbox_bottom_y = ry + rh

            ibbox_left_x = x
            ibbox_right_x = x + w
            ibbox_top_y = y
            ibbox_bottom_y = y + h

            if not (rbbox_right_x < ibbox_left_x or
                    rbbox_left_x > ibbox_right_x or
                    rbbox_top_y > ibbox_bottom_y or
                    rbbox_bottom_y < ibbox_top_y):
                return True

        return False

    def __get_coordinates(self, coord, scale):
        """This function converts the coordinates from the format [x,y,w,h] --> [x,y,x1,y1] and
        scale the coordinates, if the translated image dimension is different from the original image - using the scale
        parameter"""
        x, y, w, h = coord

        if scale:
            x_dim, y_dim = self.__original_dimension
            imagex, imagey = self.__translated_dimension
            x_scale = imagex / x_dim
            y_scale = imagey / y_dim

            x2 = int((x + w) * x_scale)
            y2 = int((y + h) * y_scale)
            x = int(x * x_scale)
            y = int(y * y_scale)

        else:
            x2 = x + w
            y2 = y + h

        _coord = (x, y, x2, y2)
        return _coord

    def __area_in_percentage(self, area):
        h, w = self.__original_dimension
        image_area = h * w
        percent = ((area / image_area) * 100)
        return percent

    def initialize(self, original_img_path, translated_img_path, annotations_path):
        """This method initialise and store within the class instance all the necessary/important details pertaining to images"""
        super().initialize(annotations_path)

        images = os.listdir(original_img_path)
        image = images[0]
        image = os.path.join(original_img_path, image)
        org_pil_image = Image.open(image)
        org_img_shape = org_pil_image.size

        translated_images = os.listdir(translated_img_path)
        translated_image = translated_images[0]
        translated_image = os.path.join(translated_img_path, translated_image)
        translated_pil_image = Image.open(translated_image)
        translated_img_shape = translated_pil_image.size

        self.__original_image_path = original_img_path
        self.__translated_image_path = translated_img_path
        self.__original_dimension = org_img_shape
        self.__translated_dimension = translated_img_shape

    def get_area(self, filename=None, lower_bound=0, upper_bound=100):
        if filename is None:
            bbox_area = super().get_area()

            h, w = self.__original_dimension
            org_area = h * w
            lower_bound = ((org_area / 100) * lower_bound)
            upper_bound = ((org_area / 100) * upper_bound)

            bbox_area = {_image: _area for _image, _area in bbox_area.items() if upper_bound >= _area >= lower_bound}

            return bbox_area

        else:
            bbox_area = super().get_area()
            return bbox_area

    def __area_available(self, image):
        area_taken = image['area']
        h, w = self.__original_dimension
        org_area = h * w
        area_available = org_area - area_taken
        return area_available

    def crop(self, crop_ped_save_path=None, random_crop_save_path=None, trans_crop_save_path=None,
             n=10, area_threshold=40, crop_random=False, translated=False, scale=False):
        """The function crops patches from images that contain pedestrians and random patches
        crop_ped_save_path parameter is the save path for cropped patches containing pedestrians
        random_crop_save_path for random patches,
        n is number of random patches to be cropped from an image, and
        area_threshold parameter is a threshold value above which no patches would be cropped
        scale parameter[bool] is to scale the bounding boxes of the translated images if the dimensions are different"""

        if translated:
            images = os.listdir(self.__translated_image_path)  # getting all the translated images
        else:
            images = os.listdir(self.__original_image_path)  # getting all the original images
        images = [image for image in images if '.png' in image]  # ignoring any files that's not an image
        images_info = self.get_image_info()  # getting information like ped_count, bbox_area, bbox_coord for all images
        bboxes = self.get_bbox_list()  # a list containing all the bbox coordinates with pedestrian in it

        for image in tqdm(images):
            image_bbox = self.get_bbox(image)  # bbox coordinate for the particular image
            if translated:
                image_path = os.path.join(self.__translated_image_path, image)  # absolute path of the translated image
            else:
                image_path = os.path.join(self.__original_image_path, image)  # absolute path of the image

            image_name = image.replace('_leftImg8bit.png',
                                       '')  # image name with no extension, key to the image_info dict

            # crop pedestrians:
            # -----------------
            pil_image = Image.open(image_path)

            i = 1  # for saving each crop of an image
            for bbox in image_bbox:
                if not self.filter_bboxes(bbox):
                    continue
                coordinates = self.__get_coordinates(bbox, scale=scale)  # converts (x,y,w,h) -> (x,y,x1,y1)
                # print(coordinates) --debugging print statement
                ped_crop = pil_image.crop(coordinates)  # crop the patch containing pedestrian
                cropped_image_name = image_name + f'_{i}.png'  # adding extension to the image name to save it
                if translated:
                    crop_path = os.path.join(trans_crop_save_path, cropped_image_name)
                else:
                    crop_path = os.path.join(crop_ped_save_path, cropped_image_name)
                ped_crop.save(crop_path)  # saving the image
                i += 1

            if crop_random:
                # crop random patches:
                # --------------------
                image_info = images_info[
                    image_name]  # grabbing the image info for the particular image from images_info dict
                area_taken = image_info['area']  # area taken by pedestrian bboxes in the image
                percentage_area = self.__area_in_percentage(
                    area_taken)  # percentage of area taken by all ped bboxes in that image
                random_bboxes = []  # a list containing random bbox cropped from an image if it has ped area< threshold
                if percentage_area <= area_threshold:  # only crop random patch if area taken by ped bbox <= a threshold percentage
                    for i in range(n):
                        intersect = True
                        while intersect:
                            random_bbox = random.choice(bboxes)
                            if self.is_intersect(random_bbox, image_bbox):
                                intersect = True
                            else:
                                random_bboxes.append(random_bbox)
                                intersect = False

                if random_bboxes:  # for the cases where the ped area is less than threshold and random_bboxes is not empty
                    i = 1
                    for _random_bbox in random_bboxes:
                        _random_bbox = self.__get_coordinates(_random_bbox)
                        random_crop = pil_image.crop(_random_bbox)
                        random_crop_name = image_name + f'_{i}.png'  # adding extension to the image name to save it
                        crop_path = os.path.join(random_crop_save_path, random_crop_name)
                        random_crop.save(crop_path)  # saving the image
                        i += 1

            # print(f'{image_name} processed.') -- debugging print statement

        print('Image cropping finished successfully')


if __name__ == '__main__':
    crop = Crop()
    crop.initialize(original_path, translated_path, ann_path)
    crop.crop(trans_crop_save_path=translated_crop_save_path, translated=True)
