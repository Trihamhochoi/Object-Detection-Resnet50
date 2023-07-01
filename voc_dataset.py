from torchvision.datasets import VOCDetection
import torch
import argparse
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomAffine, ColorJitter


def get_args():
    parser = argparse.ArgumentParser(description="Train an object detector")
    parser.add_argument("--data-path", "-d", type=str, default="../../data/pascal_voc/")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", "-b", type=int, default=4)
    parser.add_argument("--image-size", "-i", type=int, default=416)
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--log_path", "-l", type=str, default="pascal_voc")
    parser.add_argument("--save_path", "-s", type=str, default="trained_models")
    parser.add_argument("--checkpoint", "-c", type=str, default='tensorboard/animals/epoch_last.pt')  # None
    args = parser.parse_args()
    return args

# Build dataset VOC
class PASCALVOCDataset(VOCDetection):
    # build like the VOC class:
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

        # Get width and Height
        ori_width = int(data["annotation"]["size"]["width"])
        ori_height = int(data["annotation"]["size"]["height"])

        # Get bounding box and label of object
        for obj in data["annotation"]["object"]:
            bbox = obj["bndbox"]
            # Rescale the image (divide the org weight/height then multiple for assigned size)
            xmin = int(bbox["xmin"]) / ori_width * self.size
            ymin = int(bbox["ymin"]) / ori_height * self.size
            xmax = int(bbox["xmax"]) / ori_width * self.size
            ymax = int(bbox["ymax"]) / ori_height * self.size
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.classes.index(obj["name"]))

        # Transform to float for boxes and long type (int64) for label
        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)

        # combine to dictionary
        targets = {"boxes": boxes, "labels": labels}
        return image, targets


if __name__ == '__main__':
    args = get_args()

    # Data augmentation
    train_transform = Compose([
        ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
        Resize((args.image_size, args.image_size)),
        ToTensor(),
        # standard value of each chanel from Imagenet
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])
    dataset = PASCALVOCDataset(root=args.data_path, year="2007", image_set="val", download=False,
                               transform=train_transform, size=args.image_size)

    img, tg = dataset.__getitem__(23)
    print(type(img))
    print(tg)
