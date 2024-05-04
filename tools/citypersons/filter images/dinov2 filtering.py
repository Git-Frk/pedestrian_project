import os
import shutil
import json
from tqdm import tqdm

# translated_path_winter = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/translated images/winter'
translated_path_rain = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/i2i-turbo(rain)'
translated_path_night = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/i2i-turbo(night)'

ann_path = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/original images/annotations'

# -- cosine filtered list (JSON file) --
# winter_cosine_filtered_list = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/translated images/winter/winter.json'
rain_cosine_filtered_list = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/i2i-turbo(rain)/rain.json'
night_cosine_filtered_list = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/i2i-turbo(night)/night.json'

save_path = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/cosine images/cosine images(i2i-turbo)'


def filter_images(translated_path, ann_path, json_path, save_path=save_path, style=None):
    """ This function selects images in the json file, which contains images that poss the cosine similarity using DINOv2
    and place each images into corresponding folder named based on the style along with annotations for these files
    the style parameter accepts values from - ('winter', 'rain', 'night')"""

    with open(json_path) as jobject:
        filtered_list = json.load(jobject)
        filtered_list = filtered_list[style]

    img_save_path = os.path.join(save_path, style)
    ann_save_path = os.path.join(img_save_path, 'annotations')
    img_save_path = os.path.join(img_save_path, style)
    os.makedirs(img_save_path)
    os.makedirs(ann_save_path)

    filtered_annotations = [image.replace('_leftImg8bit.png','') for image in filtered_list]
    filtered_annotations = [image+'_gtBboxCityPersons.json' for image in filtered_annotations]

    print('-' * 20)
    print('Moving images:')
    for image in tqdm(filtered_list):
        image_path = os.path.join(translated_path,image)
        image_save_path = os.path.join(img_save_path,image)
        shutil.move(image_path, image_save_path)
    print(f'Moved {len(filtered_list)}')

    print('-' * 20)
    print('Copying annotations:')
    for annotation in tqdm(filtered_annotations):
        annotation_path = os.path.join(ann_path, annotation)
        annotation_save_path = os.path.join(ann_save_path,annotation)
        shutil.copyfile(annotation_path, annotation_save_path)
    print(f'Copied {len(filtered_annotations)}')


filter_images(translated_path_night, ann_path, night_cosine_filtered_list, style='night')
