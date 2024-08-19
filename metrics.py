import sys
sys.path.append('../')

from util.misc import *
import difflib
import torch

import argparse

from tqdm import tqdm

from torchmetrics.detection import MeanAveragePrecision

def get_args():
    parser = argparse.ArgumentParser(description='Calculate evaluation metrics on a specified ODVG dataset.')

    parser.add_argument('--gt_data', type=str)
    parser.add_argument('--pred_data', type=str)

    parser.add_argument('--phrase_matching', type=str, choices=['exact', 'fuzzy'], default='exact')
    parser.add_argument('--dataset_type', type=str, choices=['od', 'vg'], default='vg')


    return parser


def label_mapping(caption_list):
    uni_caption_list = list(dict.fromkeys(caption_list))
    label_map = {}
    
    for idx in range(len(uni_caption_list)):
        label_map[uni_caption_list[idx]] = idx
        
    classes = [label_map[cap] for cap in caption_list]
    
    classes = torch.tensor(classes, dtype=torch.int64)
    
    return classes, label_map


def match_phrases(pred_phrases, label_map, setting="exact"):
    all_labels = label_map.values()
    max_nonlabel = max(all_labels) + 1
    
    if setting == "exact":
        pred_labels = []
        for phrase in pred_phrases:
            try:
                label = label_map[phrase]
            except KeyError:
                label = max_nonlabel
                
            pred_labels.append(label)
        
    elif setting == "fuzzy":
        pred_labels = []
        
        for phrase in pred_phrases:
            try:
                label = label_map[phrase]
            except KeyError:
                try:
                    match = difflib.get_close_matches(phrase, label_map.keys(), cutoff=0.7)[0]
                    label = label_map[match]
                except IndexError:
                    label = max_nonlabel
                
            pred_labels.append(label)
            
    return pred_labels


def calculate_metrics(gt, pred, setting):
    assert len(gt) == len(pred), "Length of ground truth and prediction lists must be equal."
    
    all_metrics = []
    for i in tqdm(range(len(gt))):
        entry_metrics = metrics_per_img(gt, pred, i, setting)
        all_metrics.append(entry_metrics)
        
    dataset_average = calculate_averages(all_metrics)
    
    return json.dumps(dataset_average)
        
    
    
def metrics_per_img(gt, pred, i, setting):
    gts, preds = gt[i], pred[i]
    
    gt_boxes, gt_phrases = [a["bbox"] for a in gts['grounding']['regions']], [a["phrase"] for a in gts['grounding']['regions']]
    pred_boxes, pred_phrases = [a["bbox"] for a in preds['grounding']['regions']], [a["phrase"] for a in preds['grounding']['regions']]
    pred_scores = [a["logits"] for a in preds['grounding']['regions']]
    
    pred_phrases = [phrase if len(phrase) > 0 else "no_obj" for phrase in pred_phrases]

    labels_gt, label_map = label_mapping(gt_phrases)
    
    labels_pred = match_phrases(pred_phrases, label_map, setting=setting)

    
    gt_dict = [{"boxes": torch.tensor(gt_boxes),
                "labels": torch.tensor(labels_gt)}]
    
    pred_dict = [{"boxes": torch.tensor(pred_boxes),
                "labels": torch.tensor(labels_pred),
            "scores": torch.tensor(pred_scores)}]
    
    metric = MeanAveragePrecision(iou_type="bbox")
    metric.update(pred_dict, gt_dict)

    result = metric.compute()
    
    relevant_metrics = {"map": result["map"].item(),
                        "map_50": result["map_50"].item(),
                        "map_75": result["map_75"].item(),
                        "mar_1": result["mar_1"].item(),
                        "mar_10": result["mar_10"].item(),
                        "mar_100": result["mar_100"].item()}
    
    return relevant_metrics

def calculate_averages(metric_scores):

    sum_dict = {}
    avg_dict = {}

    num_entries = len(metric_scores)

    # Sum values for each metric across all dictionaries
    for metrics in metric_scores:
        for key, value in metrics.items():
            if key in sum_dict:
                sum_dict[key] += value
            else:
                sum_dict[key] = value

    # Calculate average for each metric
    for key, sum_value in sum_dict.items():
        avg_dict[key] = sum_value / num_entries
        
    return avg_dict

def calculate_metrics_od(gt, pred):
    gts = []
    preds = []

    for gt_el, pred_el in zip(gt, pred):
        gt_regions = gt_el['detection']['instances']
        pred_regions = pred_el['detection']['instances']

        gt_boxes = [a["bbox"] for a in gt_regions]
        pred_boxes = [a["bbox"] for a in pred_regions]

        gt_labels = [a["label"] for a in gt_regions]
        pred_labels = [a["label"] for a in pred_regions]

        gt_dict = {"boxes": torch.tensor(gt_boxes),
                    "labels": torch.tensor(gt_labels)}
        
        pred_dict = {"boxes": torch.tensor(pred_boxes),
                    "scores": torch.tensor([1.0]*len(pred_labels)),
                    "labels": torch.tensor(pred_labels),}
        

        gts.append(gt_dict)
        preds.append(pred_dict)

    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    metric.update(preds, gts)


    result = metric.compute()
    
    return result
    
    

if __name__ == "__main__":
    parser = get_args()
    args = parser.parse_args()

    gt = load_jsonl(args.gt_data)
    pred = load_jsonl(args.pred_data)

    if args.dataset_type == "od":
        metric_dict = calculate_metrics_od(gt, pred)
        print(metric_dict)
    else:

        metric_dict = calculate_metrics(gt, pred, args.phrase_matching)
        print(metric_dict)
        

