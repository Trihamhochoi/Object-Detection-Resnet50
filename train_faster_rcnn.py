from torchvision.datasets import VOCDetection
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomAffine, ColorJitter
import torch.nn as nn
from torchsummary import summary
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# fpn: feature pyramid network, backbone: resnet50, model: fasterRCNN
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights, \
    FasterRCNN_ResNet50_FPN_V2_Weights
import numpy as np
import cv2
import argparse
import os
import shutil
from tqdm.autonotebook import tqdm
from pprint import pprint


# use collate to follow the input of Faster RCNN model
def collate_fn(batch):
    images = []
    targets = []

    for i, t in batch:
        images.append(i)
        targets.append(t)

    return images, targets


def get_args():
    parser = argparse.ArgumentParser(description="Train an object detector")
    parser.add_argument("--data-path", "-d", type=str, default="../../data/pascal_voc/")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", "-b", type=int, default=4)
    parser.add_argument("--image-size", "-i", type=int, default=416)
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--log_path", "-l", type=str, default="pascal_voc")
    parser.add_argument("--save_path", "-s", type=str, default="trained_models")
    parser.add_argument("--checkpoint", "-c", type=str, default='./trained_models/epoch_last.pt')  #None
    args = parser.parse_args()
    return args


class PASCALVOCDataset(VOCDetection):
    def __init__(self, root, year, image_set, download, transform, size=1000):
        super().__init__(root=root, year=year, image_set=image_set, download=download, transform=transform)
        self.classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                        'train', 'tvmonitor']
        self.size = size

    def __getitem__(self, index):
        image, data = super().__getitem__(index)
        boxes = []
        labels = []
        ori_width = int(data["annotation"]["size"]["width"])
        ori_height = int(data["annotation"]["size"]["height"])

        for obj in data["annotation"]["object"]:
            bbox = obj["bndbox"]
            xmin = int(bbox["xmin"]) / ori_width * self.size
            ymin = int(bbox["ymin"]) / ori_height * self.size
            xmax = int(bbox["xmax"]) / ori_width * self.size
            ymax = int(bbox["ymax"]) / ori_height * self.size
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.classes.index(obj["name"]))
        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)
        targets = {"boxes": boxes, "labels": labels}
        return image, targets


if __name__ == '__main__':
    args = get_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    train_transform = Compose([
        ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
        Resize((args.image_size, args.image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])
    test_transform = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])
    train_set = PASCALVOCDataset(root=args.data_path, year="2007", image_set="train", download=False,
                                 transform=train_transform, size=args.image_size)
    test_set = PASCALVOCDataset(root=args.data_path, year="2007", image_set="val", download=False,
                                transform=test_transform, size=args.image_size)

    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=4,
        drop_last=True,
        shuffle=True,
        collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        num_workers=8,
        drop_last=False,
        shuffle=False,
        collate_fn=collate_fn
    )
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
                                                                 trainable_backbone_layers=0)

    # Adjust from 90+1 classes(COCO dataset) to 20+1 classes(VOC dataset) from ROI (Region of Interpret) head
    model.roi_heads.box_predictor.cls_score = nn.Linear(in_features=1024, out_features=20 + 1, bias=True)
    model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features=1024, out_features=4 * (20 + 1), bias=True)
    model.to(device)

    # create directory containing the model
    # if directory existed, remove that directory, create new one
    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    os.mkdir(args.log_path)

    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)

    # create writer and optimizer
    writer = SummaryWriter(args.log_path)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # set up checkpoint:
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    # Start to run epoch
    for epoch in range(start_epoch, args.epochs):

        # Training process
        model.train()
        train_loss = []
        train_progress_bar = tqdm(train_dataloader, colour="cyan")
        for i, (images, targets) in enumerate(train_progress_bar):
            # Run with GPU
            images = [image.to(device) for image in images]
            final_targets = [{"boxes": target["boxes"].to(device),
                              "labels": target["labels"].to(device)} for target in targets]

            # predict the output from model (4 loss function)
            output = model(images, final_targets)

            # Sum loss value
            loss_value = sum([loss for loss in output.values()])

            # Start the optimizer process
            optimizer.zero_grad()
            loss_value.backward()
            train_loss.append(loss_value.item())
            optimizer.step()
            train_progress_bar.set_description(
                "Epoch {}. Iteration {}/{} Loss {:0.4f}".format(epoch + 1, i + 1, len(train_dataloader),
                                                                np.mean(train_loss)))
            # Create metric
            writer.add_scalar("Train/Loss", np.mean(train_loss), i + epoch * len(train_dataloader))

        # START TO EVALUATE THE TEST DATALOADER with metric MAP
        model.eval()
        metric = MeanAveragePrecision(class_metrics=True)
        val_process_bar = tqdm(test_dataloader, colour='white')

        # Get the best precision score:
        best_precision = 0
        best_epoch = 0

        for i, (images, targets) in enumerate(val_process_bar):
            images = [image.to(device) for image in images]

            targets_list = [{"boxes": target["boxes"].to(device),
                             "labels": target["labels"].to(device)} for target in targets]

            # Stop the optimal process
            with torch.inference_mode():
                predictions = model(images)
            metric.update(predictions, targets_list)
            val_process_bar.set_description(
                "Validation: Iteration {}/{}".format(i + 1, len(test_dataloader))
            )

        # show MAP score
        map_dict = metric.compute()
        pprint(map_dict)
        eval_precision = map_dict['map'].item()

        # Create checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        # Save model in the last epoch we run
        torch.save(checkpoint, os.path.join(args.save_path, 'epoch_last.pt'))
        if eval_precision > best_precision:
            best_precision = eval_precision
            best_epoch = epoch + 1
            torch.save(checkpoint, os.path.join(args.save_path, 'epoch_best.pt'))

        # Early Stopping
        if epoch - best_epoch == 5:
            exit(0)
