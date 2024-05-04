import torch 
from PIL import Image
from transformers import AutoProcessor, CLIPModel
import torch.nn as nn
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

def clip_similarity(org_img, tran_img):
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)


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
    
    plt.figure(figsize = (8,6),dpi = 500) 
    
    plt.subplot(1,2,1)
    plt.imshow(image1)
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(image2)
    plt.axis('off')