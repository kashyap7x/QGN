import os
import json
from pathlib import Path


cityscapes_root = '/home/selfdriving/datasets/cityscapes_full'
train_odgt = Path('./data/train_cityscapes.odgt')
val_odgt = Path('./data/validation_cityscapes.odgt')


def make_CityScapes(mode, root, odgt):
    mask_path = os.path.join('gtFine_trainvaltest', 'gtFine', mode)
    mask_postfix = '_gtFine_labelIds.png'
    img_path = os.path.join('leftImg8bit_trainvaltest', 'leftImg8bit', mode)
    items = []
    categories = os.listdir(os.path.join(root, img_path))
    with odgt.open(mode='w+') as fo:
        for c in categories:
            c_items = [name.split('_leftImg8bit.png')[0] for name in os.listdir(os.path.join(root, img_path, c))]
            for it in c_items:
                item = {"width": 2048, "fpath_img": os.path.join(img_path, c, it + '_leftImg8bit.png'), "height": 1024, "fpath_segm": os.path.join(mask_path, c, it + mask_postfix)}
                fo.write(f'{json.dumps(item)}\n')
    
    
def main():   
    make_CityScapes('train', cityscapes_root, train_odgt)
    make_CityScapes('val', cityscapes_root, val_odgt)
        
        
if __name__ == '__main__':
    main()