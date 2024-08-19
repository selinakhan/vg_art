import json
import sys

sys.path.append("../")

import cv2
from matplotlib import pyplot as plt
import numpy as np

import torch
from torchvision.ops import box_convert

import supervision as sv
import sng_parser


def load_jsonl(path):
    ''' Load data from jsonl file. '''
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]

    return data

def save_jsonl(data, path):
    ''' Save data to jsonl file. '''
    with open(path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
            
def process_boxes(image_source, boxes, in_f="cxcywh", out_f="xyxy", absolute=True):
    h, w, _ = image_source.shape
    
    if absolute:
        boxes = boxes * torch.Tensor([w, h, w, h])
        
    xyxy = box_convert(boxes=boxes, in_fmt=in_f, out_fmt=out_f).numpy()
    
    return xyxy

def annotate(image_source, boxes, logits, phrases, raw=False, scale=0.5, thickness=1, padding=2):
    if raw:
        xyxy = process_boxes(image_source, boxes)
    else:
        xyxy = boxes
    
    xyxy = xyxy + np.array([0, 20, 0, 20])
    detections = sv.Detections(xyxy=xyxy)
    
    if len(logits) > 0:
        labels = [
            f"{phrase} {logit:.2f}"
            for phrase, logit
            in zip(phrases, logits)
        ]
    else:
        labels = phrases

    box_annotator = sv.BoxAnnotator(text_scale=scale,
                                text_thickness=thickness,
                                text_padding=padding,
                                )
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
    annotated_frame = cv2.copyMakeBorder(annotated_frame, 20, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame

def plot(boxes, logits, phrases, image, show_plot=True, raw=False, scale=0.5, thickness=1, padding=2):
    annotated_frame = annotate(image, boxes, logits, phrases, raw, scale, thickness, padding)
    
    if show_plot:
        plt.figure(figsize=(10, 10))
        plt.imshow(annotated_frame[...,::-1])
        
        plt.axis('off')
        plt.show()

    return annotated_frame
    
def parse_caption(cap, return_entities=True):
    parsed = sng_parser.parse(cap)
    
    entities = [el['span'] for el in parsed['entities']]
    str_entities = ' . '.join(entities) + ' .'

    relations = [entities[el['subject']] + ' ' + el['relation'] + ' ' + entities[el['object']] for el in parsed['relations']]
    str_relations = ' . '.join(relations) + ' .'
    
    if return_entities:
        return str_entities
    else:
        return str_relations

def preprocess_caption(caption):
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."
