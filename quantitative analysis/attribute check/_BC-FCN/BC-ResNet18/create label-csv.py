import os
import pandas as pd

ped_crop_path = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/crops/positive'
random_crop_path = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/crops/negative'

# Renaming the images(crops):
# def rename_images(fp, label = None):
#     for image in os.listdir(fp):
#         source_path = os.path.join(fp, image)
#         destination = f'{label}_{image}'
#         destination_path = os.path.join(fp,destination)
#         os.rename(source_path, destination_path)
#
#
# rename_images(ped_crop_path, 'pos')
# rename_images(random_crop_path,'neg')


ped_crops = os.listdir(ped_crop_path)
ped_crops = [image for image in ped_crops if '.png' in image]
random_crops = os.listdir(random_crop_path)
random_crops = [image for image in random_crops if '.png' in image]

ped_label = [0 for i in range(len(ped_crops))]
random_label = [1 for i in range(len(random_crops))]

ped_labels = {'x': ped_crops, 'y': ped_label}
random_labels = {'x': random_crops, 'y': random_label}

df1 = pd.DataFrame(ped_labels)
df2 = pd.DataFrame(random_labels)
df = pd.concat([df1, df2], ignore_index=True)
print(df)

# Saving the CSV file:
df.to_csv('crop.csv', index=False)
