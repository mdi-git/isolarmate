import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from ultralytics import YOLO

model = YOLO("D:/workspace/iaolarmate/00.topinfra/16_best.pt/weights/best.pt") 
results = model.train(data="D:/workspace/ultralytics/00.topinfra/labelme_json_dir_test/YOLODataset/dataset.yaml", 
                      project='D:/workspace/iaolarmate',
                      name='subfolder_name',
                      exist_ok=True,
                      patience=40, epochs=10,
                      )  
results = model.val()  


'''
# Train settings -------------------------------------------------------------------------------------------------------
model:  # path to model file, i.e. yolov8n.pt, yolov8n.yaml
data:  # path to data fil


e, i.e. i.e. coco128.yaml
epochs: 100  # number of epochs to train for
patience: 200  # epochs to wait for no observable improvement for early stopping of training
batch: 16  # number of images per batch (-1 for AutoBatch)
imgsz: 640  # size of input images as integer or w,h
save: True  # save train checkpoints and predict results
cache: False  # True/ram, disk or False. Use cache for data loading
device:  # device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
workers: 8  # number of worker threads for data loading (per RANK if DDP)
project:  # project name
name:  # experiment name
exist_ok: False  # whether to overwrite existing experiment
pretrained: False  # whether to use a pretrained model
optimizer: SGD  # optimizer to use, choices=['SGD', 'Adam', 'AdamW', 'RMSProp']
verbose: True  # whether to print verbose output
seed: 0  # random seed for reproducibility
deterministic: True  # whether to enable deterministic mode
single_cls: False  # train multi-class data as single-class
image_weights: False  # use weighted image selection for training
rect: False  # support rectangular training
cos_lr: False  # use cosine learning rate scheduler
close_mosaic: 10  # disable mosaic augmentation for final 10 epochs
resume: False  # resume training from last checkpoint
# Segmentation
overlap_mask: True  # masks should overlap during training (segment train only)
mask_ratio: 4  # mask downsample ratio (segment train only)
# Classification
dropout: 0.0  # use dropout regularization (classify train only)
'''