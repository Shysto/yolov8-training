#!/usr/bin/env python
"""Training of YOLOv8 models."""

import json
import platform
from pathlib import Path
from argparse import ArgumentParser

import torch
from easydict import EasyDict as edict
from ultralytics import YOLO, settings

from log import setup_logger
from config import *


def load_config(cfg_file: Path) -> edict:
    cfg = edict()
    try:
        with open(str(cfg_file), 'r') as f:
            cfg.update(json.load(f))
    except Exception as e:
        logger.error(f'Failed to open configuration file {str(cfg_file)}: {e}.')
        exit(1)

    return cfg

def update_config(cfg: edict, args: dict) -> edict:
    keys_to_update = []

    for k, v in args.items():
        if k in ['config', 'model', 'force_cpu', 'use_multi_gpus', 'task', 'dataset']:
            continue

        if type(v) == bool:
            if v:
                keys_to_update.append(k)
            continue

        if v is not None:
            if (k == 'pretrained') and (v.lower() in ['true', 'false']):
                args[k] = v.lower() == 'true'
            keys_to_update.append(k)

    for k in keys_to_update:
        cfg['training'][k] = args[k]

    cfg['training']['device'] = set_device(args['force_cpu'], args['use_multi_gpus'])
    cfg['task'] = args['task']
    cfg['model'] = args['model']
    cfg['dataset'] = args['dataset']

    return cfg

def check_config(cfg: edict) -> edict:
    if (type(cfg['training']['pretrained']) == str) and (not Path(cfg['training']['pretrained']).is_file()):
        logger.error(f'Failed to retrieve the pretrained model file {cfg["training"]["pretrained"]}.')
        exit(1)

    if cfg['training']['optimizer'] not in SUPPORTED_OPTIMIZERS:
        logger.error(f'Unsupported "{cfg["training"]["optimizer"]}" optimizer: supported ones are {SUPPORTED_OPTIMIZERS}.')
        exit(1)

    for k in ['workers', 'epochs', 'patience', 'imgsz', 'lr0', 'lrf', 'momentum']:
        if cfg['training'][k] <= 0:
            logger.error(f'Incorrect value for argument "{k}": expected value > 0 but got {cfg["training"][k]}.')
            exit(1)

    for k in ['batch', 'save_period']:
        if cfg['training'][k] < 1:
            cfg['training'][k] = -1

    return cfg

def get_config() -> edict:
    ap = ArgumentParser()
    ap.add_argument('-t', '--task', type=str, required=True, choices=SUPPORTED_TASKS,
        help='Task to perform.')
    ap.add_argument('-m', '--model', type=str, required=True,
        help='Model to train or path to the model to resume training from.')
    ap.add_argument('-d', '--dataset', type=str, required=True,
        help=f'Dataset folder (relative to {str(DATASET_FOLDER.absolute())}).')
    ap.add_argument('-s', '--savepath', type=str, dest='project',
        help='Path to the output directory.')
    ap.add_argument('-n', '--name', type=str,
        help='Experiment name.')
    ap.add_argument('-w', '--workers', type=int,
        help='Number of worker threads for data loading (per RANK if DDP).')
    ap.add_argument('-e', '--max_epochs', type=int, dest='epochs',
        help='Number of epochs to train for')
    ap.add_argument('--lr_initial', type=float, dest='lr0',
        help='Initial learning rate.')
    ap.add_argument('--lr_final', type=float, dest='lrf',
        help='Final learning rate (lr0 * lrf).')
    ap.add_argument('--momentum', type=float,
        help='SGD momentum/Adam beta1.')
    ap.add_argument('--patience', type=int,
        help='Number of epochs to wait for no observable improvement for early stopping of training.')
    ap.add_argument('-b', '--batch', type=int,
        help='Number of images per batch (-1 for AutoBatch).')
    ap.add_argument('--image_size', type=int, dest='imgsz',
        help='Size of input images (square).')
    ap.add_argument('--force_cpu', action='store_true',
        help='Force the use of CPU for training.')
    ap.add_argument('--use_multi_gpus', action='store_true',
        help='Whether to use multi-gpus training.')
    ap.add_argument('--save_period', type=int,
        help='Save checkpoint every x epochs (disabled if < 1).')
    ap.add_argument('--resume', action='store_true',
        help='Resume training from last checkpoint.')
    ap.add_argument('--pretrained', type=str,
        help='Whether to use a pretrained model (bool) or a model to load weights from (str).')
    ap.add_argument('--config', type=str, default=str(CONFIGS_FOLDER.joinpath('default.json')),
        help='Loads configuration from JSON file.')

    args = vars(ap.parse_args())

    cfg = load_config(args['config'])

    cfg = update_config(cfg, args)

    cfg = check_config(cfg)

    return edict(cfg)

def set_device(force_cpu: bool = False, use_multi_gpus: bool = True):
    if force_cpu:
        return 'cpu'

    if torch.cuda.is_available():
        if use_multi_gpus:
            device = list(range(torch.cuda.device_count()))
        else:
            device = [0,]
    else:
        if platform.processor() == 'arm':
            device = 'mps'
        else:
            device = 'cpu'

    return device

def get_model(task: str, model: str) -> Path:
    if Path(model).is_file():
        return Path(model)

    file = SUPPORTED_TASKS_AND_MODELS[task].get(model, None)
    if file is None:
        logger.error(f'Unsupported model {model}. Supported values are {SUPPORTED_MODELS}.')
        exit(1)

    return WEIGHTS_FOLDER.joinpath(file)

def train(cfg: edict):
    model_path = get_model(cfg.task, cfg.model)
    logger.info(f'Loading model {str(model_path)}.')
    model = YOLO(model_path, task=cfg.task)
    results = model.train(data=cfg.dataset, **cfg.training, **cfg.augmentation)

    #TODO: refactor + support validation only
    print("\n==================")
    print("Validation summary")
    print("==================\n")

    all_class_results = {
        "Classes": results.names,
        "Number of classes": results.box.nc,
        "All class metrics":
            {
                "Precision": results.box.mp,
                "Recall": results.box.mr,
                "mAP50": results.box.map50,
                "mAP75": results.box.map75,
                "mAP50-95": results.box.map,
                "Fitness": results.box.fitness()
            }
    }

    by_class_results = {
        f"Class '{results.names[class_id]}' (ID {class_id}) metrics":
            {
                _key: metric for _key, metric in zip(METRICS, results.box.class_result(i))
            }
        for i, class_id in enumerate(sorted(results.box.ap_class_index))
    }

    all_class_results.update(by_class_results)
    pretty_json = json.dumps(all_class_results, indent=4)
    print(pretty_json)


if __name__ == '__main__':
    logger = setup_logger(__name__)

    # Update Ultralytics settings
    settings.update({'weights_dir': str(WEIGHTS_FOLDER), 'datasets_dir': str(DATASET_FOLDER)})

    cfg = get_config()

    train(cfg)

    # Reset Ultralytics settings to default values
    settings.reset()

    logger.info('Script finished successfully.')
