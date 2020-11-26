from datagen.config import data_config
from collections import OrderedDict


def label_genseq_injector(objects):
    data = label_oriented_reformat(objects)
    dnew = inject_label_genseq(data)
    dlist = revert_to_list_format(dnew)
    return dlist


def entity_label(cname, scname):
    return data_config.label_type[scname] + '_' + data_config.label_name[cname]

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

def label_oriented_reformat(objects):
    data = OrderedDict({k:{'field':[], 'delimiter':[], 'value':[]} for k,v in data_config.label_name.items()})

    for idx, obj in enumerate(objects):
        cname_curr = obj['classname']
        scname_curr = obj['subclass']
        data[cname_curr][scname_curr].append(obj)

    return data

def inject_label_genseq(data):
    data_new = OrderedDict({k:{'field':[], 'delimiter':[], 'value':[]} for k,v in data_config.label_name.items()})
    genseq = 0

    for kdata, vdata in data.items():
        for kval, vval in vdata.items():
            if kval != 'delimiter':
                entity = entity_label(kdata, kval)
                if len(vval) > 0:
                    val_list = []
                    for val in vval:
                        val['label'] = entity
                        val['genseq'] = genseq
                        val_list.append(val)

                        genseq += 1
                    data_new[kdata][kval] = val_list


            else:
                label = "O"
                if len(vval)>0:
                    val = vval.copy()
                    val[0]['label'] = label
                    val[0]['genseq'] = genseq

                    data_new[kdata][kval]=[val]

                    genseq += 1
    return data_new


def revert_to_list_format(dnew):
    data_list = []
    for k,v in dnew.items():
        field = dnew[k]['field']
        delim = dnew[k]['delimiter']
        value = dnew[k]['value']
        if len(delim)>0:
            line_list = field+delim[0]+value
        else:
            line_list = field+value

        data_list += line_list
    return data_list


