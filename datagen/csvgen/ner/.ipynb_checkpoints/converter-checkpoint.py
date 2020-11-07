
import pandas as pd
from pandas.core.common import flatten
from tqdm import tqdm
from . import config
from . import utils
import random


def randomize_position_line_sequence(texts, entity):
    pairs = [(line, ent) for line, ent in zip(texts, entity)]
    pairs = utils.randomize_list(pairs)
    lines, entities = [], []
    for (lne, ent) in pairs:
        lines.append(lne)
        entities.append(ent)
    return lines, entities

def randomize_word_line_sequence(texts, entity):
    pairs = [(line, ent) for line, ent in zip(texts, entity)]
    lines, entities = [], []
    for (lns, ent) in pairs:
        assert len(lns) == len(ent), "lns==ent does not have same length"
        index_list = [idx for idx in range(len(ent))]
        random_index = utils.randomize_list(index_list)

        line_random = [lns[idx] for idx in random_index] 
        entity_random = [ent[idx] for idx in random_index] 

        lines.append(line_random)
        entities.append(entity_random)
    return lines, entities

def randomize_sentence_sequence(texts, entity):
    pairs = utils.make_pairs(texts, entity)
    random_pairs = utils.randomize_list(pairs)
    lines, entities = utils.detach_pairs(random_pairs)
    return lines, entities


def text_tag_labeling(record, flatten_list=False):
    lines, entities = [], []
    for key, value in record.items():
        if key=="provinsi" or key=="kabupaten":
            vfk = config.field_map[key]
#             print(key)
            if type(vfk) is list:
                for field_key in vfk:
                    if field_key in value:
                        vfk = field_key
#             print(vfk)
#             print(value)
            
            val = value.replace(vfk, "")
            val = val.strip().split(" ")

            field_ent = utils.field_entity_label(key, [vfk])
            value_ent = utils.value_entity_label(key, val)

            ent = field_ent + value_ent
            line = [vfk] + val

            lines.append(line)
            entities.append(ent)
        else:
            if key in config.field_map.keys():
                vfk = config.field_map[key]

                if len(vfk)>1:
                    vfk = vfk.split(" ")
                else:
                    vfk = [vfk] 

                val = str(value).strip().split(" ")

                field_ent = utils.field_entity_label(key, vfk)
                value_ent = utils.value_entity_label(key, val)

                ent = field_ent + value_ent
                line = vfk + val

                lines.append(line)
                entities.append(ent)

            else:
                val = value.strip().split(" ")
                ent = utils.value_entity_label(key, val)

                line = val
                lines.append(line)
                entities.append(ent)
    
    lines_flat = list(flatten(lines))
    entities_flat = list(flatten(entities))
    assert len(lines_flat) == len(entities_flat), f"Error, text with entities does not has same length!"
    
    if flatten_list:
        lines, entities = lines_flat, entities_flat
    
    return lines, entities


def text_tag_randomizer(texts, entity, position_prob=0.1, word_prob=0.1, sentence_prob=0.1):
    pos_random = utils.cointoss(position_prob) 
    word_random = utils.cointoss(word_prob)
    seq_random = utils.cointoss(sentence_prob)
    if pos_random:
        texts, entity = randomize_position_line_sequence(texts, entity)
    if word_random:
        texts, entity = randomize_word_line_sequence(texts, entity)
    if seq_random:
        texts, entity  = randomize_sentence_sequence(texts, entity)
        
    if not seq_random:
        pairs = utils.make_pairs(texts, entity, flat_pair=True)
        texts, entity = utils.detach_pairs(pairs)
    
    return texts, entity

def to_ner_dataframe(data, pos_prob=0.1, wrd_prob=0.1, sen_prob=0.1):
    data_dict = {
        'sentence_idx':[],
        'word': [],
        'tag': []
    }

    for idx in tqdm(range(len(data))):
        record = data.iloc[idx].to_dict()
        texts, tags = text_tag_labeling(record, flatten_list=False)        
        texts, tags = text_tag_randomizer(texts, tags, 
                                          position_prob=pos_prob, 
                                          word_prob=wrd_prob, 
                                          sentence_prob=sen_prob)
        sentence_idx = [idx for i in range(len(texts))]

        data_dict['sentence_idx'] += sentence_idx
        data_dict['word'] += texts
        data_dict['tag'] += tags
        
    return pd.DataFrame(data_dict)



def split_indices(length_indices, valid_data=0.25, test_from_valid_data=0.35):
    index_list = [i for i in range(length_indices)]
    train_length = length_indices - int(length_indices * valid_data)
    valid_length = int(length_indices * valid_data)
    test_length = int(valid_length * test_from_valid_data)
    
    train_index = random.sample(index_list, train_length)
    valtest_index =  list(set(index_list) - set(train_index))
    
    test_index = random.sample(valtest_index, test_length)
    valid_index = list(set(valtest_index) - set(test_index))
    return train_index, valid_index, test_index


def split_ner_dataframe(dframe, valid_data=0.25, test_data=0.35):
    length_sentence = len(dframe.sentence_idx.unique())
    trn_indice, val_indice, test_indice = split_indices(length_sentence, 
                                                        valid_data=valid_data,
                                                        test_from_valid_data=test_data)
    train_frame, valid_frame, test_frame = [], [], []
    for idx in tqdm(range(length_sentence)):
        sframe = dframe[dframe.sentence_idx == idx]
        if idx in trn_indice:
            train_frame.append(sframe)
        if idx in val_indice:
            valid_frame.append(sframe)
        if idx in test_indice:
            test_frame.append(sframe)
    train_frame = pd.concat(train_frame).reset_index(drop=True)
    valid_frame = pd.concat(valid_frame).reset_index(drop=True)
    test_frame = pd.concat(test_frame).reset_index(drop=True)
    return train_frame, valid_frame, test_frame


def reset_sentence_index(dframe):
    sentences_idx = dframe.sentence_idx.unique()
    for idx, sent_idx in enumerate(tqdm(sentences_idx)):
        dframe['sentence_idx'] = dframe['sentence_idx'].replace(sent_idx, idx)
    return dframe
