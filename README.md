# YOLOv8 training

## Installation



## Usage

Basic usage:
```python
python src/train.py -t <TASK> -m <MODEL> -d <DATASET>
```

The mandatory `-t` option specifies the training task (must be in `['detect', 'classify']`). <br>
More details can be found in the section [Tasks and models (weights)](#tasks-and-models).

The mandatory `-m` option specifies the model to train (must be in `['YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'YOLOv8l', 'YOLOv8x']`). <br>
More details can be found in the section [Tasks and models (weights)](#tasks-and-models).

The mandatory `-d` option specifies the trainin dataset (relative to [`./datasets`](./datasets/)). <br>
More details can be found in the section [Datasets](#datasets).

By providing the facultative `-s` option, you can specify the path to the output directory (default is `./runs/<TASK>`).

By providing the facultative `-n` option, you can specify the experiment name (default is `train`).

By providing the facultative `-w` option, you can specify the number of worker threads to use for data loading (default is 8).

By providing the facultative `-e` option, you can specify the maximum number of epochs to train for (default is 100).

By providing the facultative `-b` option, you can specify the image batch size (default is 16).

By providing the facultative `--lr_initial` option, you can specify the initial learning rate (defautl is 0.01).

By providing the facultative `--lr_final` option, you can specify the final learning rate (lr0 * lrf, default is 0.01).

By providing the facultative `--momentum` option, you can specify the SGD momentum/Adam beta1 (default is 0.937).

By providing the facultative `--patience` option, you can specify the number of epochs to wait for no observable improvement for early stopping of training (default is 50).

By providing the facultative `--image_size` option, you can specify the input image (square) size in pixels (default is 640).

By providing the facultative `--force_cpu` option, you can for the use of CPU only for training.

By providing the facultative `--use_multi_gpus` option, you can enable (if possible) multi-gpus training.

By providing the facultative `--save_period` option, you can save checkpoint every x epochs (disabled if < 1, default is -1).

By providing the facultative `--config` option, you can specify a JSON configuration file (default is [`./configs/default.json`](./configs/default.json)). <br>
More details can be found in the section [Configuration](#configuration).

ap.add_argument('-t', '--task', type=str, required=True, choices=SUPPORTED_TASKS,
        help='Task to perform.')
    ap.add_argument('-m', '--model', type=str, required=True, choices=SUPPORTED_MODELS,
        help='Model to train.')
    ap.add_argument('-d', '--dataset', type=str, required=True,
        help=f'Dataset folder (relative to {str(DATASET_FOLDER.absolute())}).')

## Tasks and models

The following training tasks are supported (values for the `-t` option):
- Detection: `'detect'`
- Classification: `'classify'`

More details about these tasks can be found on the [official website of Ultralytics](https://docs.ultralytics.com/tasks/).

The following models are supported (values for the `-m` option):
- YOLOv8 nano: `'YOLOv8n'`
- YOLOv8 small: `'YOLOv8s'`
- YOLOv8 medium: `'YOLOv8m'`
- YOLOv8 large: `'YOLOv8l'`
- YOLOv8 x-large: `'YOLOv8x'`

The model's weights will be automatically downloaded inside the folder [`./weights`](./weights/) the first time.

More details about these models can be found on the [official website of Ultralytics](https://docs.ultralytics.com/models/yolov8/).

## Datasets

### Detection task

#### Supported dataset

The following datasets are supported by default (values for the `-d` option):
- `'Argoverse.yaml'`
- `'coco.yaml'`
- `'coco8.yaml'`
- `'GlobalWheat2020.yaml'`
- `'Objects365.yaml'`
- `'open-images-v7.yaml'`
- `'SKU-110K.yaml'`
- `'VisDrone.yaml'`
- `'VOC.yaml'`
- `'xView.yaml'`

More details about these datasets can be found on the [official website of Ultralytics](https://docs.ultralytics.com/datasets/detect/).

#### Customized dataset

If you want to use your own dataset, upload it in the [`./datasets`](./datasets/) directory. <br>
The dataset structure should be:
```
my_dataset_root/
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── labels/
│       ├── img1.txt
│       ├── img2.txt
│       └── ...
└── val/
    ├── images/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    └── labels/
        ├── img1.txt
        ├── img2.txt
        └── ...
```

Labels for this format should be exported to YOLO format with one `*.txt` file per image. If there are no objects in an image, no *.txt file is required. <br>
The `*.txt` file should be formatted with one row per object in `class x_center y_center width height` format. <br>
Box coordinates must be in normalized `xywh` format (from 0 to 1). If your boxes are in pixels, you should divide `x_center` and `width` by `image_width`, and `y_center` and `height` by `image_height`. <br>
Class numbers should be zero-indexed (start with 0).
```txt
# class x_center y_center width height
0 0.5 0.5 0.2 0.2
1 0.3 0.3 0.1 0.15
...
```

You must also create a Yaml file to define the dataset root directory, the relative paths to training/validation/testing image directories, and a dictionary of class names, such as:
```yaml
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ./my_dataset_root  # dataset root dir (relative to ./datasets)
train: train/images  # train images (relative to 'path')
val: val/images  # val images (relative to 'path')
test:  # test images (optional)

# Classes
names:
  0: class_1
  1: class_2
  ...
```

### Classification task

#### Supported dataset

The following datasets (values for the `-d` option) are supported by default:
- `'caltech101'`
- `'caltech256'`
- `'cifar10'`
- `'cifar100'`
- `'fashion-mnist'`
- `'imagenet'`
- `'imagenet10'`
- `'imagenette'`
- `'imagewoof'`
- `'mnist'`

More details about these datasets can be found on the [official website of Ultralytics](https://docs.ultralytics.com/datasets/classify/).

#### Customized dataset

If you want to use your own dataset, upload it in the [`./datasets`](./datasets/) directory. <br>
The dataset structure should be:
```
dataset_root/
├── train/
│   ├── class_1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── class_2/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── ...
└── val/
    ├── class_1/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    ├── class_2/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    └── ...
```

In this folder structure, the root directory contains one subdirectory for each class in the dataset. <br>
Each subdirectory is named after the corresponding class and contains all the images for that class. <br>
Each image file is named uniquely and is typically in a common image file format such as JPEG or PNG.

## Configuration

You may tweak several parameters for the training of your model by using a customized JSON configuration file. <br>
For more details about these configuration files, the default configuration file, and a complete list of parameters that can be tweaked, please refer to the [`./configs/README.md`](./configs/README.md) file.

## Resources

- [Ultralytics - Tasks](https://docs.ultralytics.com/tasks/)
- [Ultralytics - Training](https://docs.ultralytics.com/modes/train/)
- [Ultralytics - Configuration](https://docs.ultralytics.com/usage/cfg/)
- [Ultralytics - Model](https://docs.ultralytics.com/reference/engine/model/)
- [Ultralytics - Datasets](https://docs.ultralytics.com/datasets/)
