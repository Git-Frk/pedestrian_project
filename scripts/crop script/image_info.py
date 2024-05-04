import numpy as np
import json
import os


class ImagesInfo:
    def __init__(self):
        self.__annotations_path = None
        self.__bbox_list = []
        self.__image_info = {}
        self.__pedestrian_count = None

    @staticmethod
    def filter_bboxes(coordinates, threshold_area=300):
        """This function is to discard any bounding boxes that have negative x,y coordinates and are below
        a certain area i.e., h*w"""

        x, y, w, h = coordinates
        if w * h >= threshold_area and x >= 0 and y >= 0 and x + w <= 2048 and y + h <= 1024:
            return True
        else:
            return False

    @staticmethod
    def read_annotation(filepath):
        """This is a static method to read in the annotation file"""

        with open(filepath) as jobject:
            bounding_box = json.load(jobject)
            return bounding_box

    @staticmethod
    def area_taken(coordinates):
        """this class level function calculates the area taken by a single bbox"""

        area = 0
        for coord in coordinates:
            h, w = coord[-2], coord[-1]
            a = h * w
            area += a
        return area

    def initialize(self, anno_path, bbox_parameter='bbox'):
        """This method reads in the annotation file for each image and stores the relevant information in
        the instance attribute, which can be retrieved using the method - get_image_info()"""

        self.__annotations_path = anno_path

        files = os.listdir(self.__annotations_path)

        count = []

        for file in files:
            imageinfo = {}
            filepath = os.path.join(self.__annotations_path, file)

            # print(filepath) -- debug print statement

            bounding_box = self.read_annotation(filepath)
            objects = bounding_box['objects']
            ped_coords = [obj[bbox_parameter] for obj in objects if obj['label'] != 'ignore']

            # populating the instance attribute with relevant information pertaining to each image
            n = len(ped_coords)
            count.append(n)
            imageinfo['count'] = n

            area = self.area_taken(ped_coords)
            imageinfo['area'] = area

            imageinfo['bboxes'] = ped_coords
            for ped_coord in ped_coords:
                if self.filter_bboxes(ped_coord):
                    self.__bbox_list.append(ped_coord)

            key = file.replace('_gtBboxCityPersons.json', '')
            self.__image_info[key] = imageinfo

        self.__pedestrian_count = np.array(list(count))

    def get_image_info(self):
        """This method retrieves the information pertaining to the pedestrians for each image in the dataset"""
        return self.__image_info

    def get_ped_count(self, filename=None):
        """This method return the total number of pedestrians in all images in the dataset if the filename
        parameter is not specified, if specified pedestrian count for that image will be returned"""

        count_list = {file: count['count'] for file, count in self.__image_info.items()}

        if filename is None:
            return count_list
        else:
            return count_list[filename]

    def get_area(self, filename=None, threshold=0):
        """This method returns a dictionary containing file name and the area taken by all pedestrian in that image."""

        area_list = {file: info['area'] for file, info in self.__image_info.items()}
        if filename is None:
            return area_list
        else:
            return area_list[filename]

    def get_bbox(self, filename=None):
        """This method returns a dictionary with filename and the bounding box information for the entire dataset
        if filename parameter is not specified. If specified returns a list of all coordinates for that particular image."""

        bbox_list = {file: count['bboxes'] for file, count in self.__image_info.items()}
        if filename is None:
            return bbox_list
        else:
            key = filename.replace('_leftImg8bit.png', '')
            return bbox_list[key]

    def max_count(self):
        return self.__pedestrian_count.max()

    def min_count(self):
        return self.__pedestrian_count.min()

    def total_count(self):
        return np.sum(self.__pedestrian_count)

    def average(self):
        return self.__pedestrian_count.mean()

    def count_above_theta(self, theta):
        """This method is to identify images having certain number(theta parameter) of pedestrians or more in it.
        The only purpose of the method is to investigate and identify certain parameters for cropping, if required """

        filt_list = {file: count['count'] for file, count in self.__image_info.items() if count['count'] > theta}
        if len(filt_list) > 0:
            print(f'There are {len(filt_list)} images with pedestrian count more than {theta}')
        else:
            print(f'There are no images in the dataset with pedestrian count more than {theta}')
        return filt_list

    def get_bbox_list(self):
        """This method return the list that holds all the bounding boxes that satisfied the required criterion"""

        return self.__bbox_list


if __name__ == '__main__':
    bbox_path = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/annotations'

    image_info = ImagesInfo()

    image_info.initialize(bbox_path)
