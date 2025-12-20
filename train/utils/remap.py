import json

json_path = "datasets/AgriNet-1K/wds_folds/class_map.json"
txt_path = "datasets/AgriNet-1K/AgriNet.txt"

new_txt_path = "datasets/AgriNet-1K/AgriNet-wds.txt"

with open(json_path, 'r', encoding='utf-8') as f:
    class_map = json.load(f)

idx_to_cls = {v: k for k, v in class_map.items()}

with open(txt_path, 'r', encoding='utf-8') as f:
    cls_to_name = {}
    idx_to_name = {}
    for line in f.readlines():
        cls, name = line.strip().split('\t')
        class_name = name.strip()
        
        cls_to_name[cls] = class_name
        idx_to_name[class_map[cls]] = class_name

idx_to_name = dict(sorted(idx_to_name.items()))

with open(new_txt_path, 'w', encoding='utf-8') as f:
    for idx, name in idx_to_cls.items():
        f.write(f"{name}\n")