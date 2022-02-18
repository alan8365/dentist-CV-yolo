import json
import os
import glob

from random import shuffle
from shutil import copyfile

project_path = os.path.dirname(os.path.abspath(__file__))

dataset_dir_path = os.path.join(project_path, 'datasets')
main_dir_path = os.path.join(dataset_dir_path, 'pano630')
main_sub_dir_path = {
    'train': os.path.join(main_dir_path, 'train'),
    'val': os.path.join(main_dir_path, 'val')
}

source_dir_path = os.path.join(dataset_dir_path, 'source', 'PANO_1_630')


def convert(w, h, min_x, min_y, max_x, max_y):
    dw = 1. / w
    dh = 1. / h

    x = (min_x + max_x) / 2.0
    y = (min_y + max_y) / 2.0
    w = max_x - min_x
    h = max_y - min_y

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [x, y, w, h]


def json_to_yolo(file_path, dir_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result_file_path = os.path.join(dir_path, 'labels', os.path.basename(file_path).replace('.json', '.txt'))

    with open(result_file_path, 'w') as result_file:
        for datum in data['shapes']:
            pos = datum['points']
            pos_x = (pos[0][0], pos[1][0])
            pos_y = (pos[0][1], pos[1][1])

            min_x, max_x = min(pos_x), max(pos_x)
            min_y, max_y = min(pos_y), max(pos_y)
            x, y, w, h = convert(data['imageWidth'], data['imageHeight'], min_x, min_y, max_x, max_y)

            yolo_tooth_type = tooth_type_to_id[datum['label']]
            result_file.write(f'{yolo_tooth_type} {x} {y} {w} {h}\n')

            tooth_type_count_dict[datum['label']] += 1


if __name__ == "__main__":
    tooth_type_to_id = {'13': 0, '23': 1, '33': 2, '43': 3}
    tooth_type_count_dict = {key: 0 for key in tooth_type_to_id.keys()}

    if not os.path.isdir(main_dir_path):
        os.mkdir(main_dir_path)

        os.mkdir(main_sub_dir_path['train'])
        os.mkdir(os.path.join(main_sub_dir_path['train'], 'images'))
        os.mkdir(os.path.join(main_sub_dir_path['train'], 'labels'))

        os.mkdir(main_sub_dir_path['val'])
        os.mkdir(os.path.join(main_sub_dir_path['val'], 'images'))
        os.mkdir(os.path.join(main_sub_dir_path['val'], 'labels'))

        print(f'mkdir: {main_dir_path}')

    if len(glob.glob(os.path.join(main_dir_path, 'val', '**', '*.jpg'))) == 0:
        imgs = glob.glob(os.path.join(source_dir_path, '**', '*.jpg'), recursive=True)
        shuffle(imgs)

        split_point = len(imgs) // 5 * 4
        split_imgs = {
            'train': imgs[:split_point],
            'val': imgs[split_point:]
        }

        for dir_type in ('train', 'val'):
            for img in split_imgs[dir_type]:
                dist = os.path.join(main_sub_dir_path[dir_type], 'images', os.path.basename(img))

                copyfile(img, dist)

                # wld_img = weber(cv2.imread(img, 0))
                # cv2.imwrite(dist, wld_img * 255)

                json_file = img.replace('.jpg', '.json')
                json_dist = os.path.join(main_sub_dir_path[dir_type], 'label', os.path.basename(json_file))

                # Some image have not json if it doesn't contain target tooth.
                if os.path.isfile(json_file):
                    json_to_yolo(json_file, main_sub_dir_path[dir_type])

        print('Tooth type count:')
        for key, value in tooth_type_count_dict.items():
            print(f'{key}: {value}')
    else:
        print('Image already in folder.')

    for dir_type in ('train', 'val'):
        with open(os.path.join(main_dir_path, f'{dir_type}.txt'), 'w') as f:
            imgs = glob.glob(os.path.join(main_sub_dir_path[dir_type], '**', '*.jpg'), recursive=True)

            for img in imgs:
                f.write(f'{img}\n')
