import json
import os


def coco2flickr(path, output_path):
    count = 0
    with open(os.path.join(output_path, 'captions.token.src'), 'w', encoding='utf-8') as of:
        for name in os.listdir(path):
            with open(os.path.join(output_path, name + '.split'), 'w', encoding='utf-8') as sf:
                with open(os.path.join(path, name), 'r', encoding='utf-8') as f:
                    js = json.load(f)
                    id_filename = dict()
                    for info in js['images']:
                        filename = info['file_name']
                        sf.write(filename + '\n')
                        image_idx = info['id']
                        id_filename[image_idx] = filename
                    for anno in js['annotations']:
                        image_idx = anno['image_id']
                        # 一句话内换行符！！要处理掉
                        cap = anno['caption'].replace("\r", "").replace("\n", "")
                        of.write('{}#0\t{}\n'.format(id_filename[image_idx], cap))
                        count += 1

            print('cur:{}'.format(count))


if __name__ == '__main__':
    coco2flickr(r'F:\Dataset\annotations', r'F:\Dataset')
