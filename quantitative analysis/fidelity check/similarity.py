import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch.nn as nn
import os
import json
from tqdm import tqdm
import re

original_image_path = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/original images/ images'
translated_image_path = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/i2i-turbo(night)'


def cosine_similarity(original_image, translated_image):
    # this is helper function computes the cosine similarity of the original image and translated image dinov2
    # embeddings

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

    image1 = Image.open(original_image)
    with torch.no_grad():
        inputs1 = processor(images=image1, return_tensors="pt").to(device)
        outputs1 = model(**inputs1)  # the inputs1 is a dictionary and hence the unpacking
        image_features1 = outputs1.last_hidden_state
        image_features1 = image_features1.mean(dim=1)  # converting image_features1 tensor to a vector

    image2 = Image.open(translated_image)
    with torch.no_grad():
        inputs2 = processor(images=image2, return_tensors="pt").to(device)
        outputs2 = model(**inputs2)
        image_features2 = outputs2.last_hidden_state
        image_features2 = image_features2.mean(dim=1)

    cos = nn.CosineSimilarity(dim=0)
    sim = cos(image_features1[0], image_features2[0]).item()
    sim = (sim + 1) / 2  # Adding 1 to the cosine similarity shifts the range of values from [-1, 1] to [0, 2] and
    # dividing the adjusted similarity score by 2 scales the range of values back to [0, 1].

    return sim


def check_fidelity(original_images, translated_images, condition, threshold=None, save_files=True):
    # this is the main function that compares and computes the similarity between the original and the translated image

    if threshold is None and condition == 'winter':
        threshold = 0.86
    elif threshold is None and condition == 'rain':
        threshold = 0.87
    elif threshold is None and condition == 'night':
        threshold = 0.90

    print(f'The adverse condition is {condition} and the threshold value is {threshold}')

    filtered_images = []

    images = os.listdir(translated_images)
    org_images = os.listdir(original_images)

    for image in tqdm(images):

        if '.png' not in image:
            continue

        if condition == 'winter':
            img_name = re.sub(r'w_leftImg8bit\.png$', '', image)
        elif condition == 'rain':
            img_name = re.sub(r'r_leftImg8bit\.png$', '', image)
        elif condition == 'night':
            img_name = re.sub(r'n_leftImg8bit\.png$', '', image)

        for img in org_images:
            if img_name in img:
                orig_image = img
                break

        org_image = os.path.join(original_images, orig_image)
        trans_image = os.path.join(translated_images, image)

        cos_sim = cosine_similarity(org_image, trans_image)

        # print(f'Cosine Similarity: {cos_sim}')

        if cos_sim >= threshold:
            filtered_images.append(image)

    print(f'{len(images) - len(filtered_images)} images discarded from the translated images')

    if save_files:
        filtered_files = {condition: filtered_images}
        # save_path = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/translated images/winter'
        filename = f'{condition}.json'
        filename = os.path.join(translated_image_path, filename)
        with open(filename, 'w') as jobject:
            json.dump(filtered_files, jobject)
            print(f'filtered images filenames saved to {filename}')

    return filtered_images


if __name__ == '__main__':
    hq_images = check_fidelity(original_image_path, translated_image_path, threshold=0.95, condition='rain')
    # print(hq_images)

# ----------------------------------------Original Code-------------------------------------------------------
#
# import torch
# from transformers import AutoImageProcessor, AutoModel
# from PIL import Image
# import torch.nn as nn
# import os
# import json
#
# original_image_path = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/original images'
# translated_image_path = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/translated images/rain_and_night/winter'
#
#
# def cosine_similarity(original_image, translated_image):
#     # this is helper function computes the cosine similarity of the original image and translated image dinov2
#     # embeddings
#
#     device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
#
#     processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
#     model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
#
#     image1 = Image.open(original_image)
#     with torch.no_grad():
#         inputs1 = processor(images=image1, return_tensors="pt").to(device)
#         outputs1 = model(**inputs1)  # the inputs1 is a dictionary and hence the unpacking
#         image_features1 = outputs1.last_hidden_state
#         image_features1 = image_features1.mean(dim=1)  # converting image_features1 tensor to a vector
#
#     image2 = Image.open(translated_image)
#     with torch.no_grad():
#         inputs2 = processor(images=image2, return_tensors="pt").to(device)
#         outputs2 = model(**inputs2)
#         image_features2 = outputs2.last_hidden_state
#         image_features2 = image_features2.mean(dim=1)
#
#     cos = nn.CosineSimilarity(dim=0)
#     sim = cos(image_features1[0], image_features2[0]).item()
#     sim = (sim + 1) / 2  # Adding 1 to the cosine similarity shifts the range of values from [-1, 1] to [0, 2] and
#     # dividing the adjusted similarity score by 2 scales the range of values back to [0, 1].
#
#     return sim
#
#
# def check_fidelity(original_images, translated_images, condition, threshold=None, save_files=True):
#     # this is the main function that compares and computes the similarity between the original and the translated image
#
#     if threshold is None and condition == 'winter':
#         threshold = 0.86
#     elif threshold is None and condition == 'rain':
#         threshold = 0.87
#     elif threshold is None and condition == 'night':
#         threshold = 0.90
#
#     print(f'The adverse condition is {condition} and the threshold value is {threshold}')
#
#     filtered_images = []
#
#     images = os.listdir(translated_images)
#
#     for image in images:
#
#         if '.png' not in image:
#             continue
#
#         org_image = os.path.join(original_images, image)
#         trans_image = os.path.join(translated_images, image)
#
#         cos_sim = cosine_similarity(org_image, trans_image)
#
#         # print(f'Cosine Similarity: {cos_sim}')
#
#         if cos_sim >= threshold:
#             filtered_images.append(image)
#
#     print(f'{len(images) - len(filtered_images)} images discarded from the translated images')
#
#     if save_files:
#         filtered_files = {condition: filtered_images}
#         # save_path = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/translated images/winter'
#         filename = f'{condition}.json'
#         filename = os.path.join(translated_image_path, filename)
#         with open(filename, 'w') as jobject:
#             json.dump(filtered_files, jobject)
#             print(f'filtered images filenames saved to {filename}')
#
#     return filtered_images
#
#
# if __name__ == '__main__':
#     hq_images = check_fidelity(original_image_path, translated_image_path, threshold=90, condition='winter')
#     # print(hq_images)
