from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO("D:/workspace/iaolarmate/00.topinfra/16_best.pt/weights/best.pt")
results = model.predict(source="D:\workspace\\test_image_naju",
                        line_thickness=2, save= False, show = True, classes = [4],
                        hide_conf = True, hide_labels = False
                        # conf=0.4
                        ) 

# Prediction settings --------------------------------------------------------------------------------------------------
# source:  # source directory for images or videos
# show: False  # show results if possible
# save_txt: False  # save results as .txt file
# save_conf: False  # save results with confidence scores
# save_crop: False  # save cropped images with results
# hide_labels: False  # hide labels
# hide_conf: False  # hide confidence scores

# vid_stride: 1  # video frame-rate stride
# line_thickness: 3  # bounding box thickness (pixels)
# visualize: False  # visualize model features
# augment: False  # apply image augmentation to prediction sources
# agnostic_nms: False  # class-agnostic NMS
# classes:  # filter results by class, i.e. class=0, or class=[0,2,3]
# retina_masks: False  # use high-resolution segmentation masks
# boxes: True # Show boxes in segmentation predictions