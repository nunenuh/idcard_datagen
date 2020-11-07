import random
from . import config

def field_entity_label(key, text):
    label_map = config.entity_map[key]
    label = config.entity_prefix_map["field"] + "_" + label_map
    entity = bilou_prefixer(text, label=label)
    return entity

def value_entity_label(key, text):
    label_map = config.entity_map[key]
    label = config.entity_prefix_map["value"] + "_" + label_map
    entity = bilou_prefixer(text, label=label)
    return entity


def bilou_prefixer(text_list, label=None):
    out = []
    text_len = len(text_list)
    if text_len==1:
        bl = "U"
        if label!=None: bl =  bl + "-" + label
        out.append(bl)
    elif text_len>1:
        for idx, text in enumerate(text_list):
            if idx==0: 
                bl = "B"
                if label!=None: bl = bl + "-" + label
                out.append(bl)
            elif idx < text_len - 1: 
                bl = "I"
                if label!=None: bl = bl + "-" + label
                out.append(bl)
            else: 
                bl = "L"
                if label!=None: bl =  bl + "-" + label
                out.append(bl)
    return out

def cointoss(true_prob=0.5):
    data = [True, False]
    weights = [true_prob, 1-true_prob]
    result = random.choices(data, weights)
    return result[0]

def flatten_pairs(data):
    lines, entities = [], []
    for (text, entity) in data:
        lines += text
        entities += entity
    return lines, entities

def make_pairs(texts, entity, flat_pair=True):
    pairs = [(line, ent) for line, ent in zip(texts, entity)]
    if flat_pair:
        lines, entities = flatten_pairs(pairs)
        pairs = [(t, e) for t, e in zip(lines, entities)]
    
    return pairs

def detach_pairs(pairs):
    lines, entities = [], []
    for (text, entity) in pairs:
        lines.append(text)
        entities.append(entity)
    return lines, entities


def randomize_list(data):
    result = sorted(data, key = lambda x: random.random())
    return result
