import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

# --------------------Train Model --------------------
image_path = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/crops'
YOLO_model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)


def train_model(fp, epochs, model=YOLO_model):
    _results = model.train(data=fp, epochs=epochs, imgsz=32)

    return _results


# --------------------Plotting Accuracy and Loss --------------------
results_path = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/quantitative analysis/binary classifier/BC-YOLO/runs/classify/train-100 epochs/results.csv'


def plot_results(fp):
    results = pd.read_csv(results_path)

    plt.figure()
    plt.plot(results['                  epoch'], results['             train/loss'], label='train loss')
    plt.plot(results['                  epoch'], results['               val/loss'], label='val loss', c='red')
    plt.grid()
    plt.title('Loss vs epochs')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend()

    plt.figure()
    plt.plot(results['                  epoch'], results['  metrics/accuracy_top1'] * 100)
    plt.grid()
    plt.title('Validation accuracy vs epochs')
    plt.ylabel('accuracy (%)')
    plt.xlabel('epochs')

    plt.show()


# -------------------- Prediction --------------------
weights = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/quantitative analysis/binary classifier/BC-YOLO/runs/classify/train-50 epochs/weights/best.pt'


# test_image = ''

def predict(weights, fp):
    model = YOLO(weights)  # load a custom model

    results = model(fp)  # predict on an image
    # print(results)

    names_dict = results[0].names

    probs = results[0].probs.data.tolist()

    print(names_dict)
    print(probs)

    print(names_dict[np.argmax(probs)])


# --------------------Main Function ------------------
def run(mode, **kwargs):
    if mode == 'train':
        epochs = kwargs['epochs']
        train_model(image_path, epochs=epochs, model=YOLO_model)
    elif mode == 'plot':
        fp = kwargs['fp']
        plot_results(fp)
    elif mode == 'predict':
        weights = kwargs['weights']
        fp = kwargs['fp']
        predict(weights, fp)


val_path = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/quantitative analysis/binary classifier/BC-YOLO/val'
for image in os.listdir(val_path):
    print(f'The image: {image}')
    test_image = os.path.join(val_path, image)
    run('predict', weights=weights, fp=test_image)
    print('-' * 20)
# run('plot', fp = results_path)
