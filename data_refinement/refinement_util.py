import numpy as np
import networkx as nx
from statistics import mean 

import difflib

import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchvision.ops import box_convert, box_iou


from more_itertools import consecutive_groups
from collections import defaultdict

import re

from postprocess_refinement import *

def remove_empty_preds(entry):
    nonzero = np.where(np.array([len(phrase) for phrase in entry['phrases']]) <= 1)[0]
    mask = torch.ones(entry['boxes'].shape[0], dtype=torch.bool)
    mask[nonzero] = False
    new_phrases = [p for p in entry['phrases'] if len(p) > 1]
    
    entry['boxes'] = entry['boxes'][mask]
    entry['max_logits'] = entry['max_logits'][mask]
    entry['phrases'] = new_phrases
    entry['logits'] = entry['logits'][mask]
    
    return entry


def unmerge_phrase_groups(entry, tokenized, tokenizer, text_threshold=0.3):
    ''' Determine which prases are a result of multiple concatenated phrases due to thresholding.
        Pick the phrase group with highest mean probability. '''
    
    logits = entry['logits']
    phrases = entry['phrases']
        
    # Determine phrase groups 
    phrase_groups = []
    for l in logits:
        phrase_group = [*map(list, consecutive_groups(torch.where(l > text_threshold)[0].tolist()))]
        phrase_groups.append(phrase_group)

    updated_phrases = phrases.copy()
    
    # Filter phrases of individual boxes with multiple concatenated phrases
    for i, a in enumerate(phrase_groups):
        if len(a) == 1:
            continue
        else: # Box is comprised of multiple phrases
            avg_probs = [mean(logits[i][el].tolist()) for el in a]
            max_idx = avg_probs.index(max(avg_probs))

            for idx, el in enumerate(a):
                if idx != max_idx:
                    logits[i][el] = 0 # Zero out the logits of the other phrases
                    
            new_phrase = get_phrases_from_posmap(logits[i] > text_threshold, tokenized, tokenizer)
            updated_phrases[i] = new_phrase

    assert len(phrases) == len(updated_phrases), "Length of phrases must be the same."
    
    entry['phrases'] = updated_phrases
    entry['logits'] = logits
    
    return entry

def cc_match(exact_match):
    pairs = torch.stack(exact_match).T

    G = nx.Graph()
    edges = [tuple(x) for x in pairs.tolist()]
    G.add_edges_from(edges)

    # Find all connected components, each component is a set of connected nodes
    connected_components = list(nx.connected_components(G))

    # Convert components to tensor and sort them
    output = [sorted(list(component)) for component in connected_components]
    return output

def exact_matches_intra_img(bboxes, threshold=0.8):
    ''' This function takes into account more than two matches. '''
    
    bboxes = box_convert(boxes=bboxes, in_fmt="cxcywh", out_fmt="xyxy")

    g_box_iou = box_iou(bboxes, bboxes).fill_diagonal_(0) # Fill diagonal to filter out self-matches
    exact_match = torch.where(g_box_iou > threshold)

    return cc_match(exact_match)

def remove(idx, boxes, max_logits, phrases, logits):
    # Mask for the rows to keep
    mask = torch.ones(boxes.size(0), dtype=torch.bool)
    mask[idx] = False
    
    updated_phrases = [p for i, p in enumerate(phrases) if i not in idx]
    
    return boxes[mask], max_logits[mask], updated_phrases, logits[mask]

def phrase_update_intra_img(logits, text_threshold=0.3):
    ''' 
    1) For each entry in the matches, pick the most likely phrase if there are multiple phrases concatenated. 
    2) Between all matches, pick the most likely phrase based on the mean probability of their logits. 
    '''
    
    # Determine phrase groups for each matched box
    phrase_groups = []
    for l in logits:
        phrase_group = [*map(list, consecutive_groups(torch.where(l > text_threshold)[0].tolist()))]
        phrase_groups.append(phrase_group)
            
    # Calculate mean probability of each (updated) phrase group
    phrase_probs = [mean(logits[i][a].tolist()) if len(logits[i]) > 0 else logits[i] for i, a in enumerate(phrase_groups)]
    preferred_phrase = phrase_probs.index(max(phrase_probs))

    return preferred_phrase

def remove_intra_img_redundant_matches(matches, entry, text_threshold=0.3):
    ''' Remove redundant matches from the same image. '''
    
    logits, boxes, max_logits, phrases = entry['logits'], entry['boxes'], entry['max_logits'], entry['phrases']

    remove_ids = []
    for match in matches:
        match_logits = [logits[m] for m in match]
        best_match_idx = phrase_update_intra_img(match_logits, text_threshold)
    
        remove_entry = match[:best_match_idx] + match[best_match_idx+1:]
        
        for el in remove_entry:
            remove_ids.append(el)
            
    if len(remove_ids) > 0:
        updated_boxes, updated_max_logits, updated_phrases, updated_logits = remove(remove_ids, boxes, max_logits, phrases, logits)
        
        entry['boxes'] = updated_boxes
        entry['max_logits'] = updated_max_logits
        entry['phrases'] = updated_phrases
        entry['logits'] = updated_logits
    
    return entry
    
    
def exact_matches_inter_img(ent_boxes, cap_boxes):
    ''' This function only takes into account two matches.'''
    
    # print(ent_boxes)
    ent_boxes = box_convert(boxes=ent_boxes, in_fmt="cxcywh", out_fmt="xyxy")
    cap_boxes = box_convert(boxes=cap_boxes, in_fmt="cxcywh", out_fmt="xyxy")

    g_box_iou = box_iou(ent_boxes, cap_boxes)
    exact_match = torch.where(g_box_iou > 0.8)
    pairs = torch.stack(exact_match).T
    
    return pairs # left: ent, right: cap

def phrase_update_inter_img(entities, captions, cap_threshold, ent_threshold):
    ''' 
    1) For each entry in the matches, pick the most likely phrase if there are multiple phrases concatenated. 
    2) Between all matches, pick the most likely phrase based on the mean probability of their logits. 
    
    '''

    ent_phrase_groups = [*map(list, consecutive_groups(torch.where(entities > ent_threshold)[0].tolist()))]
    cap_phrase_groups = [*map(list, consecutive_groups(torch.where(captions > cap_threshold)[0].tolist()))]

    ent_phrase_probs = [mean(entities[g].tolist()) if len(g) > 0 else entities[g] for g in ent_phrase_groups]
    cap_phrase_probs = [mean(captions[gr].tolist()) if len(gr) > 0 else captions[gr] for gr in cap_phrase_groups]

    max_ent_idx = ent_phrase_probs.index(max(ent_phrase_probs))
    max_cap_idx = cap_phrase_probs.index(max(cap_phrase_probs))


    favorite_probs = [ent_phrase_probs[max_ent_idx], cap_phrase_probs[max_cap_idx]]
    
    max_idx = favorite_probs.index(max(favorite_probs))
    
    return max_idx # max_idx 0 : entiy, 1 : caption


def remove_inter_img_redundant_matches(matches, cap_entry, ent_entry, cap_threshold, ent_threshold):
    ''' Remove redundant matches between caption and entity input predictions. '''
    
    ent_remove, cap_remove = [], []
    ent_logits = ent_entry['logits']
    cap_logits = cap_entry['logits']
    
    for match in matches:
        match_idx = phrase_update_inter_img(ent_logits[match[0]], cap_logits[match[1]],
                                            cap_threshold, ent_threshold)
        
        if match_idx == 0:
            cap_remove.append(match[1].item())
        else:
            ent_remove.append(match[0].item())
            
    if len(ent_remove) > 0:
        updated_ent_boxes, updated_ent_max_logits, updated_ent_phrases, updated_ent_logits = remove(ent_remove, ent_entry['boxes'], ent_entry['max_logits'], ent_entry['phrases'], ent_logits)
        ent_entry['boxes'] = updated_ent_boxes
        ent_entry['max_logits'] = updated_ent_max_logits
        ent_entry['phrases'] = updated_ent_phrases
        ent_entry['logits'] = updated_ent_logits
        
    if len(cap_remove) > 0:
        updated_cap_boxes, updated_cap_max_logits, updated_cap_phrases, updated_cap_logits = remove(cap_remove, cap_entry['boxes'], cap_entry['max_logits'], cap_entry['phrases'], cap_logits)
        cap_entry['boxes'] = updated_cap_boxes
        cap_entry['max_logits'] = updated_cap_max_logits
        cap_entry['phrases'] = updated_cap_phrases
        cap_entry['logits'] = updated_cap_logits
        
    return cap_entry, ent_entry

def merge(ent_entry, cap_entry):
    
    ''' Merge both sets. Text logits are omited here, as they do not align. '''
    
    assert ent_entry['boxes'].shape[0] == ent_entry['max_logits'].shape[0] == ent_entry['logits'].shape[0] == len(ent_entry['phrases'])
    assert cap_entry['boxes'].shape[0] == cap_entry['max_logits'].shape[0] == cap_entry['logits'].shape[0] == len(cap_entry['phrases'])

    merged_boxes = torch.cat((ent_entry['boxes'], cap_entry['boxes']), dim=0)
    merged_max_logits = torch.cat((ent_entry['max_logits'], cap_entry['max_logits']), dim=0)

    merged_phrases = ent_entry['phrases'] + cap_entry['phrases']

    return merged_boxes, merged_max_logits, merged_phrases 

def is_within_sides(boxes1, boxes2, margin=0.05, matches=2):
      '''
      Check if boxes in boxes1 are strictly within boxes in boxes2
      by aligning on two or more sides with a margin.
      
      '''
      
      # Check alignment of each side with margins
      left_matches = (torch.abs(boxes1[:, None, 0] - boxes2[:, 0]) <= margin).long()
      right_matches = (torch.abs(boxes1[:, None, 2] - boxes2[:, 2]) <= margin).long()
      top_matches = (torch.abs(boxes1[:, None, 1] - boxes2[:, 1]) <= margin).long()
      bottom_matches = (torch.abs(boxes1[:, None, 3] - boxes2[:, 3]) <= margin).long()

      # Sum how many sides match
      sides_matching = left_matches + right_matches + top_matches + bottom_matches

      # Require at least #matches sides to match closely
      is_within_strict = sides_matching >= matches

      return is_within_strict
  
def is_within(boxes1, boxes2, margin=0.025, area_diff_pct=75):
    # Check spatial containment with margins
    is_within = (boxes1[:, None, 0] >= boxes2[:, 0] - margin) & \
                (boxes1[:, None, 1] >= boxes2[:, 1] - margin) & \
                (boxes1[:, None, 2] <= boxes2[:, 2] + margin) & \
                (boxes1[:, None, 3] <= boxes2[:, 3] + margin)

    return is_within

def remove_post(idx, boxes, logits, phrases):
      mask = torch.ones(boxes.size(0), dtype=torch.bool)
      mask[idx] = False

      updated_phrases = [p for i, p in enumerate(phrases) if i not in idx]

      return boxes[mask], logits[mask], updated_phrases
  
  
def simple_phrase_removal(phrases):
    ''' Remove phrases that are just articles and numbers. '''
    
    updated_phrases = []
    removed_phrases = []
    articles = ['a', 'an', 'the']
    numbers = ['one', 'two', 'three', 'four', 'five', 'six',
               'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve']
    
    for i, phrase in enumerate(phrases):
        split_phrase = phrase.split(' ')
        if split_phrase[0] in articles:
            phrase = ' '.join(split_phrase[1:])
            
        split_phrase = phrase.split(' ')
        if split_phrase[-1] in articles:
            phrase = ' '.join(split_phrase[:-1])
            
        split_phrase = phrase.split(' ')
        if split_phrase[0] in numbers:
            phrase = ' '.join(split_phrase[1:])
            
        split_phrase = phrase.split(' ')
        if split_phrase[-1] in numbers:
            phrase = ' '.join(split_phrase[:-1])
            
        if "#" in phrase:
            phrase = re.sub('[#]', '', phrase)
            
        if len(phrase) == 0:
            removed_phrases.append(i)
            
        updated_phrases.append(phrase)
                    
    return updated_phrases, removed_phrases

def merged_postprocessing(boxes, logits, phrases, pil_img):
    ''' Final postprocessing removing encased boxes with the same label.'''
    
    # Remove boxes that cover the whole image
    # large_boxes = (boxes[:, 2] >= 0.95) & (boxes[:, 3] >= 0.95)
    # large_box_indices = torch.nonzero(large_boxes, as_tuple=True)[0] 
    # boxes, logits, phrases = remove_post(large_box_indices, boxes, logits, phrases)

    # Remove boxes with just articles and numbers
    phrases, remove_idx = simple_phrase_removal(phrases)
    if len(remove_idx) > 0:
       boxes, logits, phrases = remove_post(remove_idx, boxes, logits, phrases)
       
    # Convert boxes for easier iou calculation.
    boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")

    # First match on phrase
    indices_dict = defaultdict(list)
    for i, e in enumerate(phrases):
        indices_dict[e].append(i)
        
    phrase_groups = dict(indices_dict)
    
    # First filter encased boxes with same label
    remove_idxs = []
    for phrase in phrase_groups:
        if len(phrase_groups[phrase]) > 1: # Boxes with the same phrase
            p_boxes = boxes[phrase_groups[phrase]]
            p_logits = logits[phrase_groups[phrase]]

            encased_boxes = is_within(p_boxes, p_boxes).fill_diagonal_(0)
            encased_idx = torch.where(encased_boxes == True)
            matches = torch.stack(encased_idx).T
            
            if len(matches) == 0: # none encased
                continue
            
            for match in matches:
                match_t = torch.tensor(match)
                encased_match_ids = torch.tensor(phrase_groups[phrase])[torch.tensor(match_t)]
                encased_logits = p_logits[match].tolist()

                best_idx = encased_logits.index(max(encased_logits))
                
                enc_group = list(map(phrase_groups[phrase].__getitem__, match)) 
                remove = enc_group[:best_idx] + enc_group[best_idx+1:]

                remove_idxs.extend(remove)
        else:
            continue
        
    boxes, logits, phrases = remove_post(list(set(remove_idxs)), boxes, logits, phrases) 
    
    boxes = box_convert(boxes=boxes, in_fmt="xyxy", out_fmt="cxcywh") # Do this before returning
    
    return boxes, logits, phrases
    
def exact_matches_cap_group(entry, threshold, tokenizer):
    matches = exact_matches_intra_img(entry['boxes'])
    
    remove_ids = []
    for match in matches:
        logits = [entry['logits'][m] for m in match]
        
        # Determine phrase groups for each matched box
        phrase_groups = []
        for l in logits:
            phrase_group = [*map(list, consecutive_groups(torch.where(l > threshold)[0].tolist()))]
            phrase_groups.append(phrase_group)
        
        inbetweens = []
        for i, g in enumerate(phrase_groups):
            if i == 0:
                prev = g[-1][-1]
                continue
            
            cur = g[0][0]
            inbetween = list(range(prev, cur))[1:]
            inbetweens.append(inbetween)
            
            prev = g[-1][-1]
        
        
        for j, inbet in enumerate(inbetweens):
            try:
                tokens = entry['tokenized']['input_ids'][torch.tensor(inbet)]
                if tokens in [2004, 1997]:
                    # print(phrase_groups[j], inbet, phrase_groups[j+1])
                    entry['logits'][match[j]][inbet] = 1
                    entry['logits'][match[j]][phrase_groups[j+1]] = 1
                    
                    phrase = get_phrases_from_posmap(entry['logits'][match[j]] > threshold, 
                                                                         entry['tokenized'], 
                                                                         tokenizer)
                    entry['phrases'][match[j]] = phrase
                    remove_ids.append(match[j+1])
                    
            except: # It's not a single word
                continue
        
    entry['boxes'], entry['max_logits'], entry['phrases'], entry['logits'] = remove(remove_ids, 
                                                                                    entry['boxes'], 
                                                                                    entry['max_logits'], 
                                                                                    entry['phrases'], 
                                                                                    entry['logits'])
    return entry
    

def initial_clean(entry):
    ''' Remove inital predictions we don't want. '''
    
    to_remove = ['print', 'japanese print', 'this print', 'the print',
                 'prints', 'japanese tryptich print', 'a print', 'woodblock prints',
                 'a japanese print', 'woodblock print',
                 'portrait', 'a portrait', 'full - length portrait', 
                 'half - length portrait', 'a full - length portrait',
                 'portraits', 'bust portrait', 'a bust portrait',
                 'image', 'an image', 'the image', 'images',
                 'scene', 'a scene', 'scenes', 'the scene',
                 'selection', 'a selection', 'shows', 'showing']
    
    remove_idxs = [i for i, e in enumerate(entry['phrases']) if e in to_remove]
    remove_words = [e for e in entry['phrases'] if e in to_remove]
    
    if len(remove_idxs) > 0:
        entry['boxes'], entry['max_logits'], entry['phrases'], entry['logits'] = remove(remove_idxs,
                                                                                        entry['boxes'],
                                                                                        entry['max_logits'],
                                                                                        entry['phrases'],
                                                                                        entry['logits'])
        
    return entry
    

def clean_and_merge(output_cap,
                    cap_threshold,
                    tokenizer, pil_img):
    
    ''' Clean and merge boxes and phrases.'''
    
    # tokenized_ent = output_ent['tokenized']
    tokenized_cap = output_cap['tokenized']
    
    # output_ent = initial_clean(output_ent)
    output_cap = initial_clean(output_cap)
    
    # Un-merge phrase groups.
    output_cap = unmerge_phrase_groups(output_cap, tokenized_cap, tokenizer, text_threshold=cap_threshold)

    # Exact location match in caption prediction with grouped phrases.
    output_cap = exact_matches_cap_group(output_cap, cap_threshold, tokenizer)
    
    
    # Exact location max within same image. 
    matches_cap = exact_matches_intra_img(output_cap['boxes'])
    
    output_cap = remove_intra_img_redundant_matches(matches_cap,
                                                    output_cap,
                                                    text_threshold=cap_threshold)
    
    boxes, max_logits, phrases = output_cap['boxes'], output_cap['max_logits'], output_cap['phrases']
    
    # Final postprocessing.
    boxes, max_logits, phrases = merged_postprocessing(boxes, max_logits, phrases, pil_img)
    
    return boxes, max_logits, phrases
    
################### OBJECT DETECTION ######################

def phrase_2_id(phrases, label_map, dataset_name):
    ''' Convert phrase to label id. '''

    pred_labels = []
    new_phrases = []

    remove = []
    
    label_map = {y.lower(): int(x) for x, y in label_map.items()}
    
    if dataset_name == 'iconart':
        for i, phrase in enumerate(phrases):

            if phrase == 'mother mary':
                phrase = 'mary'
                new_phrases.append(phrase)
            else:
                new_phrases.append(phrase)

            try:
                label = label_map[phrase]

            except KeyError:
                try:
                    match = difflib.get_close_matches(phrase, label_map.keys(), cutoff=0.8)[0]
                    label = label_map[match]
                except IndexError:
                    label = -1
                    remove.append(i)
                
            pred_labels.append(label)

    if dataset_name == 'artdl':
        for i, phrase in enumerate(phrases):
            if 'saint' not in phrase:
                phrase = 'saint ' + phrase
                new_phrases.append(phrase)
            else:
                new_phrases.append(phrase)
            
            try:
                label = label_map[phrase]
            
            except KeyError:
                try:
                    match = difflib.get_close_matches(phrase, label_map.keys(), cutoff=0.7)[0]
                    label = label_map[match]
                except IndexError:
                    label = -1
                    remove.append(i)

            pred_labels.append(label)

    
    return pred_labels, new_phrases, remove

def remove_post_od(idx, boxes, logits, phrases, pred_labels):
      mask = torch.ones(boxes.size(0), dtype=torch.bool)
      mask[idx] = False

      updated_phrases = [p for i, p in enumerate(phrases) if i not in idx]
      updated_labels = [l for i, l in enumerate(pred_labels) if i not in idx]

      return boxes[mask], logits[mask], updated_phrases, updated_labels


def single_class(logits, phrases):
    remove = []

    # First match on phrase
    indices_dict = defaultdict(list)
    for i, e in enumerate(phrases):
        indices_dict[e].append(i)
        
    phrase_groups = dict(indices_dict)


    for phrase in phrase_groups:
        cur = phrase_groups[phrase]
        max_logits = [logits[i].item() for i in cur]

        best = max_logits.index(max(max_logits))
        
        remove_entry = cur[:best] +  cur[best+1:]

        remove.extend(remove_entry)

    return remove


def clean_and_merge_od(output, tokenizer, pil_img, label_map, dataset_name):

    tokenized = output['tokenized']

    if dataset_name == 'iconart':

        output = unmerge_phrase_groups(output, tokenized, tokenizer, text_threshold=0.25)

        matches = exact_matches_intra_img(output['boxes'])

        output = remove_intra_img_redundant_matches(matches,
                                                    output,
                                                    text_threshold=0.25)


    elif dataset_name == 'artdl':
        output = unmerge_phrase_groups(output, tokenized, tokenizer, text_threshold=0.25)

        matches = exact_matches_intra_img(output['boxes'])

        output = remove_intra_img_redundant_matches(matches,
                                                    output,
                                                    text_threshold=0.25)
        
        # Ensure single occurence of phrases
        remove = single_class(output['max_logits'], output['phrases'])

        output['boxes'], output['max_logits'], output['phrases'] = remove_post(remove, output['boxes'], output['max_logits'], output['phrases'])


        
    
    boxes, max_logits, phrases = output['boxes'], output['max_logits'], output['phrases']

    boxes, max_logits, phrases = merged_postprocessing(boxes, max_logits, phrases, pil_img)

    pred_labels, new_phrases, remove = phrase_2_id(phrases, label_map, dataset_name)

    boxes, max_logits, phrases, pred_labels = remove_post_od(remove, boxes, max_logits, new_phrases, pred_labels)

    return boxes, max_logits, phrases, pred_labels
        






    