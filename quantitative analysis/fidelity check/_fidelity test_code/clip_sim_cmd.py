import torch 
from PIL import Image
from transformers import AutoProcessor, CLIPModel
import torch.nn as nn
import argparse


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

parser = argparse.ArgumentParser()
parser.add_argument('org_img', metavar = 'original image', type = str, help = 'path to orginal image')
parser.add_argument('tran_img', metavar = 'translated image', type = str, help = 'path to translated image')
args = parser.parse_args()

org_img = args.org_img
tran_img = args.tran_img


image1 = Image.open(org_img)
with torch.no_grad():
    inputs1 = processor(images=image1, return_tensors="pt").to(device)
    image_features1 = model.get_image_features(**inputs1)
    
image2 = Image.open(tran_img)
with torch.no_grad():
    inputs2 = processor(images=image2, return_tensors="pt").to(device)
    image_features2 = model.get_image_features(**inputs2)    
    
cos = nn.CosineSimilarity(dim=0)
sim = cos(image_features1[0],image_features2[0]).item()
sim = (sim+1)/2
print('Similarity:', sim)