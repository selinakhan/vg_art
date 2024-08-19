from util.misc import load_jsonl, save_jsonl

from PIL import Image


from tqdm import tqdm
import numpy as np

import torch
from torchvision.ops import box_convert

from GroundingDINO.groundingdino.util.inference import load_model
import GroundingDINO.groundingdino.datasets.transforms as T

import argparse
import sng_parser

import copy



def get_args():
    parser = argparse.ArgumentParser(description='Run inference on a specified ODVG dataset.')

    parser.add_argument('--data_path', type=str)
    parser.add_argument('--cap_style', type=str, default='full', choices=['full', 'entities', 'relations'])
    parser.add_argument('--dataset_type', type=str, choices=['VG', 'OD'], default='VG')

    parser.add_argument('--img_path', type=str)

    parser.add_argument('--model_checkpoint_path', type=str)
    parser.add_argument('--model_config_path', type=str)

    parser.add_argument('--text_threshold', type=float, default=0.25)
    parser.add_argument('--box_threshold', type=float, default=0.25)

    parser.add_argument('--label_mapping', type=str, default=None)
    parser.add_argument('--output_path', type=str)


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
    
def parse_caption(cap, cap_style):
    if cap_style == 'full':
        return cap
       
    else:
        parsed = sng_parser.parse(cap)
        
        entities = [el['span'] for el in parsed['entities']]
        str_entities = ' . '.join(entities) + ' .'

        relations = [entities[el['subject']] + ' ' + el['relation'] + ' ' + entities[el['object']] for el in parsed['relations']]
        str_relations = ' . '.join(relations) + ' .'
        
        if cap_style == 'entities':
            return str_entities
        elif cap_style == 'relations':
            return str_relations

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
    
def process_boxes(image_source, boxes, in_f="cxcywh", out_f="xyxy", absolute=True):
    h, w, _ = image_source.shape
    
    if absolute:
        boxes = boxes * torch.Tensor([w, h, w, h])
        
    xyxy = box_convert(boxes=boxes, in_fmt=in_f, out_fmt=out_f).numpy()

    return xyxy

def predict(model, image, caption, device, box_threshold, text_threshold):

    image = image.to(device)
    
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)
    pred_features = outputs["box_feats"].cpu()[0]  # pred_features.shape = (nq, 256)

    mask = prediction_logits.max(dim=1)[0] > box_threshold

    logits = prediction_logits[mask]  # logits.shape = (n, 256)
    boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)
    features = pred_features[mask]  # features.shape = (n, 256)
    
    phrases = [
        get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
        for logit
        in logits
    ]
    
    output = {'boxes': boxes, 'max_logits': logits.max(dim=1)[0], 
              'phrases': phrases, 'logits': logits,
              'features': features,
              'caption': caption, 'tokenized': tokenized}
        
    return output

def run_inference(dataset, img_path, cap_style, model, 
                  text_threshold, box_threshold, device,
                  dataset_type, label_mapping=None):
    
    ''' Run inference on GroundingDINO.'''
    
    dataset_after_inference = []
    
    for i, entry in enumerate(tqdm(dataset)):
        new_entry = copy.deepcopy(entry)
        
        image = img_path + entry['filename']
        pil_img, image_transformed = load_image(image)
        
        if dataset_type == "VG":
            caption = entry['grounding']['caption']
            
            caption_processed = preprocess_caption(caption)
            caption_parsed = parse_caption(caption_processed, cap_style)
            

            output = predict(
                model=model,
                image=image_transformed,
                caption=caption_parsed,
                device=device,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )
        
            boxes = process_boxes(pil_img, output['boxes'])
            
            new_entry['grounding']['regions'] = [{"bbox": box.tolist(), 
                                                  "phrase": phrase,
                                                  "logits": logit.item(),
                                                  "features": feature.tolist()} for box, phrase, logit, feature in zip(boxes, output['phrases'], output['max_logits'], output['features'])]

        elif dataset_type == "OD":
            caption = 0
            
            caption =  [el['category'] for el in entry['detection']['instances']]

            caption = ' . '.join(caption) + ' .'
            
            caption_processed = preprocess_caption(caption)
            

            output = predict(
                model=model,
                image=image_transformed,
                caption=caption_processed,
                device=device,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )
        
            boxes = process_boxes(pil_img, output['boxes'])
            
            new_entry['detection']['instances'] = [{"bbox": box.tolist(), 
                                                  "label": phrase,
                                                  "logits": logit.item()} for box, phrase, logit in zip(boxes, output['phrases'], output['max_logits'])]

            

        dataset_after_inference.append(new_entry)

    return dataset_after_inference

if __name__ == "__main__":
    parser = get_args()
    args = parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    dataset = load_jsonl(args.data_path)
    
    model = load_model(args.model_config_path, 
                        args.model_checkpoint_path, 
                        device=device)
    
    model = model.to(device)
    model.eval()

    dataset_after_inference = run_inference(dataset, 
                                            args.img_path, 
                                            args.cap_style,
                                            model, 
                                            float(args.text_threshold), 
                                            float(args.box_threshold), 
                                            device,
                                            args.dataset_type,
                                            args.label_mapping)

    save_jsonl(dataset_after_inference, args.output_path)
    