import os
import re
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import decimal
import numpy as np
import pandas as pd
from ultralytics import YOLO

YOLO_model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

weights = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/quantitative analysis/binary classifier/BC-YOLO/runs/classify/train-50 epochs/weights/best.pt'

image_path = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/crops/i2i-turbo crops'


def predict(pre_trained_weights, fp):
    model = YOLO(pre_trained_weights)  # load a custom model

    results = model(fp)  # predict on an image

    names_dict = results[0].names
    probs = results[0].probs.data.tolist()
    # print(names_dict)
    # print(probs)
    prediction = names_dict[np.argmax(probs)]

    return prediction


# predict(pre_trained_weights=weights, fp=image_path)

def select_best_images(crops_path, style=None, save=False):
    image_crops_path = os.path.join(crops_path, style)
    crops = os.listdir(image_crops_path)
    crops_series = pd.Series(crops)
    selected_images = []  # list of selected images

    while len(crops_series) > 0:
        img_name = re.sub(r'_\d+\.png$', '', crops_series[0])

        filt = crops_series.str.contains(img_name)
        image_crops = list(crops_series[filt].values)  # a list to hold all crops from a single image
        crops_in_image = len(image_crops)
        crops_series = crops_series[~filt]  # removing crops of image being processed
        crops_series.reset_index(drop=True, inplace=True)

        # print(f'Processing {img_name}', '-' * 10)
        count = 0
        for crop in image_crops:
            crop_path = os.path.join(image_crops_path, crop)
            prediction = predict(weights, crop_path)
            if prediction == 'positive':
                count += 1

        if crops_in_image < 10:
            if count >= int(decimal.Decimal(count / 2).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP)):
                selected_images.append(img_name)
        else:
            if count >= round(crops_in_image * 0.6):
                selected_images.append(img_name)

    print(f'{len(selected_images)} images were selected for augmentation')
    selected_images = [image + '_leftImg8bit.png' for image in selected_images]

    if save:
        selected_images_dict = {style: selected_images}
        filename = style + '.json'
        save_path = os.path.join(crops_path, filename)

        with open(save_path, 'w') as json_file:
            json.dump(selected_images_dict, json_file)

        print(f'{style}.json file saved successfully')

    return selected_images


if __name__ == '__main__':
    filtered_image_list = select_best_images(image_path, style='night', save=True)
