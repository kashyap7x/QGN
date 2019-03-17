python train.py --id 'cityscapes' --list_train './data/train_cityscapes.odgt' --list_val './data/validation_cityscapes.odgt' --root_dataset '../../datasets/cityscapes_raw/' --num_class 19 --transform_dict '{"0": -1, "1": -1, "2": -1, "3": -1, "4": -1, "5": -1, "6": -1, "7": 0, "8": 1, "9": -1, "10": -1, "11": 2, "12": 3, "13": 4, "14": -1, "15": -1, "16": -1, "17": 5, "18": -1, "19": 6, "20": 7, "21": 8, "22": 9, "23": 10, "24": 11, "25": 12, "26": 13, "27": 14, "28": 15, "29": -1, "30": -1, "31": 16, "32": 17, "33": 18}' --imgSize 768 896 1024 1152 1280 --imgMaxSize 2560 --batch_size_per_gpu 1 --arch_decoder QGN_resnet34 --deep_sup_scale 1.0
