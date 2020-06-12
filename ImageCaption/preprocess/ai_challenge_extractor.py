import json
import pkuseg

JSON_FILE = 'D:/caption_train_annotations_20170902.json'
OUTPUT = 'D:/output.token'

length_limit = 33

seg = pkuseg.pkuseg()
used_file = []

with open(OUTPUT, 'w', encoding='utf-8') as o:
    with open(JSON_FILE, 'r', encoding='utf-8') as rd:
        js = json.load(rd)
        for idx, info in enumerate(js):
            name = info['image_id']
            used = False
            for i, cap in enumerate(info['caption']):
                anno = cap.replace("\r", "").replace("\n", "").replace("、", "")
                cut = seg.cut(anno)
                if len(cut) > length_limit:
                    continue
                used = True
                anno = ' '.join(cut)
                o.write('{}#{}\t{}\n'.format(name, i, anno))
            if used:
                used_file.append(name)
            if idx > 0 and idx % 1000 == 0:
                print(idx)

print(len(used_file))
with open(OUTPUT + ".list", 'w', encoding='utf-8') as f:
    for name in used_file:
        f.write(name + '\n')

if __name__ == '__main__':
    pass
