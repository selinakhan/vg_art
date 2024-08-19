from refinement_util import *
import json
from PIL import Image

from groundingdino.util.inference import load_model
from tqdm import tqdm

import torch
import groundingdino.datasets.transforms as T

import argparse
import sng_parser

import copy

import sys
sys.path.append('../')
from util.misc import *

def get_args():
    parser = argparse.ArgumentParser(description='Run iterative data refinement.')

    parser.add_argument('--data_path', type=str)
    parser.add_argument('--label_map', type=str)
    parser.add_argument('--dataset_type', type=str, choices=['VG', 'OD'], default='OD')
    parser.add_argument('--dataset_name', type=str, choices=['ukiyoe', 'iconart', 'artdl'], default='ukiyoe')

    parser.add_argument('--save_path', type=str)
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--model_config_path', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--model_config_path', type=str)
    parser.add_argument('--model_path', type=str)

    return parser
   
    
def load_image(path):

    transform = T.Compose(
    [
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
    image_source = Image.open(path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed


def preprocess_caption(caption):
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."

def get_phrases_from_posmap(posmap, tokenized, tokenizer, left_idx=0, right_idx=255):
    assert isinstance(posmap, torch.Tensor), "posmap must be torch.Tensor"
    if posmap.dim() == 1:
        posmap[0: left_idx + 1] = False
        posmap[right_idx:] = False
        non_zero_idx = posmap.nonzero(as_tuple=True)[0].tolist()
        
        token_ids = [tokenized["input_ids"][i] for i in non_zero_idx]
        
        return tokenizer.decode(token_ids)
    else:
        raise NotImplementedError("posmap must be 1-dim")

def predict(model, image, caption, box_threshold, text_threshold):
    
    model = model.to(device)
    image = image.to(device)
    
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)

    
    mask = prediction_logits.max(dim=1)[0] > box_threshold

    logits = prediction_logits[mask]  # logits.shape = (n, 256)
    boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)
    
    phrases = [
        get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '') #TODO
        for logit
        in logits
    ]
    
    output = {'boxes': boxes, 'max_logits': logits.max(dim=1)[0], 
              'phrases': phrases, 'logits': logits,
              'caption': caption, 'tokenized': tokenized}
        
    return output

                 
def iterative_refinement(model, dataset, img_path):
    '''
    Postprocess GroundingDINO Inference for iterative data refinement. 
    
    - dataset: jsonl file.
    
    '''
    
    threshold = 0.20
    
    refined_dataset = []
    
    for i, entry in enumerate(tqdm(dataset)):
        image = img_path + entry['filename']
        pil_img, image_transformed = load_image(image)
        
        caption = entry['grounding']['caption']
        
        caption_processed = preprocess_caption(caption)

        print('Caption:', caption_processed)
    
        
        output_cap = predict(
            model=model,
            image=image_transformed,
            caption=caption_processed,
            box_threshold=threshold,
            text_threshold=threshold
        )
        
        boxes, logits_max, phrases = clean_and_merge(output_cap,
                                                    threshold,
                                                    model.tokenizer,
                                                    pil_img)
        

        
    
        new_entry = entry.copy()
        new_boxes = process_boxes(pil_img, boxes, in_f="cxcywh", out_f="xyxy", absolute=True)

        new_entry['grounding']['regions'] = [{"bbox": box.tolist(), "phrase": phrase} for box, phrase in zip(new_boxes, phrases)]

        refined_dataset.append(new_entry)
    
    return refined_dataset

def iterative_refinement_od(model, dataset, img_path, label_map, dataset_name):
    
    refined_dataset = []
    
    for i, entry in enumerate(tqdm(dataset)):
        image = img_path + entry['filename']
        pil_img, image_transformed = load_image(image)
        
        caption =  ['mother ' + el['category'] if el['category'] == 'Mary' else el['category'] for el in entry['detection']['instances']]

        caption = ' and '.join(caption) + ' .'

        caption_processed = preprocess_caption(caption)
    
        
        output = predict(
            model=model,
            image=image_transformed,
            caption=caption_processed,
            box_threshold=0.25,
            text_threshold=0.25
        )


        boxes, logits_max, phrases, cat_ids = clean_and_merge_od(output,
                                                    model.tokenizer,
                                                    pil_img,
                                                    label_map,
                                                    dataset_name)

        
        new_entry = copy.deepcopy(entry)
        new_boxes = process_boxes(pil_img, boxes)
        
        new_entry['detection']['instances'] = [{"bbox": box.tolist(), 
                                                "category": label_map[str(label)],
                                                "label": label} for box, phrase, label in zip(new_boxes, phrases, cat_ids)]

        
        refined_dataset.append(new_entry)
    
    return refined_dataset


if __name__ == "__main__":
    parser = get_args()
    args = parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    dataset = load_jsonl(args.data_path)

    torch.cuda.empty_cache()
    model = load_model(args.model_config_path, 
                        args.model_path, 
                        device=device)
    
    if args.dataset_type == "VG":
        refined_dataset = iterative_refinement(model, dataset, args.img_path)
        
        save_jsonl(refined_dataset, args.save_path)

    else:
        label_map = json.load(open(args.label_map, 'r'))

        refined_dataset = iterative_refinement_od(model, dataset, args.img_path, label_map, args.dataset_name)
        save_jsonl(refined_dataset, args.save_path)
    
    
    
    
