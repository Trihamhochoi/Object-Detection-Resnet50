# Object-Detection-Resnet50

## Introduction:
By using model Resnet-50 which was built by myself, I used Pytorch to detect image of 20 classes.

Here is my pytorch implementation of the model described in the [RESNET paper](https://arxiv.org/abs/1512.03385). 

`Notes:` By using transfer learning method, I adjusted from 90 classes(COCO dataset) to 20 classes(VOC dataset) from ROI (Region of Interpret) head

``` python
model.roi_heads.box_predictor.cls_score = nn.Linear(in_features=1024, out_features=20 + 1, bias=True)
model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features=1024, out_features=4 * (20 + 1), bias=True)
```

## Dataset:
### Statistics of datasets: 

it was gotten [VOC main page](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit) 

**20 classes:**
- Person: person
- Animal: bird, cat, cow, dog, horse, sheep
- Vehicle: aeroplane, bicycle, boat, bus, car, motorbike, train
- Indoor: bottle, chair, dining table, potted plant, sofa, tv/monitor

The train/val data has 11,530 images containing 27,450 ROI annotated objects and 6,929 segmentations.

### New developments:
Size of segmentation dataset substantially increased.

People in action classification dataset are additionally annotated with a reference point on the body.

## Settings:
For optimizer and learning rate, I use:
- **SGD** optimizer with different learning rates (1e-3 in most cases).

Additionally,I will set up 100 epochs (using early stopping if after 5 epochs, if there is not greater score, it will stop train proccess) ,which is seen as a loop over `batch_size: 4` since this is a vast model, if I set 4 batch size, it will be suitable with my latop configuration. 

## Training:

If you want to train a model with default parameters, you could run:

python train_animal.py 
If you want to adjust your preference parameters, here is some option you can choose:
| Parameters | Abbreviation | Default | Description |
|:---------:|:---------------:|:---------:|:---------:|
|    --batch-size |    -b  |    4                                   |Select suitable batch size|
|    --data-path  |    -d  |    'directory/contain/yourdata'        |directory contains dataset|
|    --lr         |        |    1e-3                                |modify learning rate|
|    --image-size |   -i  |    416                                  |scale image to 416x416|
|    --epochs     |    -e  |    100                                 |modify epoch number|
|    --log-path   |    -l  |    pascal_voc                          |[directory contains metrics visualization](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)|
|    --save-path  |    -s  |    trained_models                      |[Please download my model I trained on my driver](https://drive.google.com/file/d/1GTAnXmVJMaoGO6Kj0Pjf2ZEIKmKy1WWp/view?usp=sharing)|
|    --checkpoint |   -sc  |    /trained_models/epoch_last.pt   |directory which saves the train model|


`For example:` python train_faster_rcnn.py -p dataset_location --log-patch directory/contain/visualization --data-path directory/contain/yourdata
 
 **How to view tensorboard:** 
```
 tensorboard --logdir directory/contain/visualization
```
## Evaluating:

You could preview my evaluating process throughout Tenserboard. After 22 epochs, I reached:
- The highest accuracy score: 0.7 in epoch 19
  ![Accuracy score](./demo/run-epoch.png "Accuracy score")

- Confusion matrix:
 ![Confusion matrix](./demo/confusion-matrix.png "Confusion matrix")

- Cross entropy loss in 2 processes:
 ![Loss plot](./demo/plot-loss.png "Loss plot")

## Testing:

After buiding models, I started to implement the testing process by  file [test_cnn.py](.code/test_cnn.py)

Regarding to the above confusion matrix, this model had good prediction in 4 classes: spider, chicken, horse and butterfly.

In contrast, it scored the worst prediction in class: Cat. 
