import argparse
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomAffine, ColorJitter
from voc_dataset import PASCALVOCDataset
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
import torch.nn as nn
import cv2
from PIL import Image
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="Train an object detector")
    parser.add_argument("--data-path", "-d", type=str, default="../../data/pascal_voc/")
    parser.add_argument("--image-size", "-is", type=int, default=416)
    parser.add_argument("--model", '-m', type=str, default='trained_models/fasterrcnn_best.pt')
    parser.add_argument("--image", '-i', type=str, default='./test_img/1.jpg')
    parser.add_argument("--threshold", '-t', type=str, default=0.8)
    args = parser.parse_args()
    return args


args = get_args()

# SET UP MODEL, FOLDER AND PARAMETER BEFORE TRAINING AND TESTING PROCESS

# Set device is CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Build model
model = fasterrcnn_resnet50_fpn(weights=None)
model.roi_heads.box_predictor.cls_score = nn.Linear(in_features=1024, out_features=20 + 1, bias=True)
model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features=1024, out_features=4 * (20 + 1), bias=True)
model.to(device)

# get checkpoint:
checkpoint = torch.load(args.model)
model.load_state_dict(checkpoint['model_state_dict'])

# Build dataset and get classes
# Transformer
transformer = Compose([
    ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
    Resize((args.image_size, args.image_size)),
    ToTensor(),
    # standard value of each chanel from Imagenet
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])

dataset = PASCALVOCDataset(root=args.data_path, year="2007", image_set="val", download=False,
                           transform=transformer, size=args.image_size)
classes = dataset.classes
print(classes)


if __name__ == '__main__':
    # Get test img and convert to Tensor
    url = './test_img/2.jpg'
    org_img = cv2.imread(url)
    # Get width and height
    width = org_img.shape[1]
    height = org_img.shape[0]

    # transform img
    pil_img = Image.open(url)
    img = transformer(pil_img)
    img = torch.unsqueeze(img, dim=0)
    img = img.to(device)

    # TESTING: predict the image according to model
    model.eval()
    with torch.no_grad():
        prediction = model(img)

    prediction = prediction[0]
    boxes = prediction['boxes']
    labels = prediction['labels']
    scores = prediction['scores']

    # Get object has scores > threshold
    final_boxes = []
    final_labels = []
    final_scores = []
    # Get Final pred
    for b,l,s in zip(boxes,labels,scores):
        if s > args.threshold:
            final_boxes.append(b)
            final_labels.append(l)
            final_scores.append(s)

    for b,l,s in zip(final_boxes,final_labels,final_scores):
        # define x y and rescale
        xmin, ymin, xmax, ymax = b
        xmin = int(xmin / args.image_size * width)
        xmax = int(xmax / args.image_size * width)
        ymin = int(ymin / args.image_size * height)
        ymax = int(ymax / args.image_size * height)

        # draw bounding box and label text
        final_img = cv2.rectangle(org_img,(xmin, ymin), (xmax,ymax),(255,0,0), 1)

        final_img = cv2.putText(img=final_img ,
                                text=classes[l]+'{:.2f}'.format(s.item()),
                                org=(xmin, ymin+20),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1,
                                color=(0,0,128),
                                thickness=2,
                                lineType=cv2.LINE_AA)

    cv2.imwrite("output.jpg",final_img)