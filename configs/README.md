# Configuration files

You can customize your training using a JSON configuration file. <br>
By default, the [`default.json`](./default.json) configuration is used. <br>
Parameters not defined by your customized configration file will be set to their default value. <br>

The JSON file structure must be:
```json
{
    "augmentation": {
        "parameter": "value",
        ...
    },
    "training": {
        "parameter": "value",
        ...
    },
    "validation": {
        "parameter": "value",
        ...
    }
}
```
where the meaning of the parameters for each section is described in this document.

## Augmentation

The list of possible parameters for this section:

| Parameter name | Type  | Default value | Explanation                                     |
|----------------|-------|---------------|-------------------------------------------------|
| hsv_h          | float | 0.015         | Image HSV-Hue augmentation (fraction)           |
| hsv_s          | float | 0.7           | Image HSV-Saturation augmentation (fraction)    |
| hsv_v          | float | 0.4           | Image HSV-Value augmentation (fraction)         |
| degrees        | float | 0.0           | Image rotation (+/- deg)                        |
| translate      | float | 0.1           | Image translation (+/- fraction)                |
| scale          | float | 0.5           | Image scale (+/- gain)                          |
| shear          | float | 0.0           | Image shear (+/- deg)                           |
| perspective    | float | 0.0           | Image perspective (+/- fraction), range 0-0.001 |
| flipud         | float | 0.0           | Image flip up-down (probability)                |
| fliplr         | float | 0.5           | Image flip left-right (probability)             |
| mosaic         | float | 1.0           | Image mosaic (probability)                      |
| mixup          | float | 0.0           | Image mixup (probability)                       |
| copy_paste     | float | 0.0           | Segment copy-paste (probability)                |

## Training

The list of possible parameters for this section:

| Parameter name  | Type                    | Default value | Explanation                                                                       |
|-----------------|-------------------------|---------------|-----------------------------------------------------------------------------------|
| epochs          | int                     | 100           | Number of epochs to train for                                                     |
| patience        | int                     | 50            | Epochs to wait for no observable improvement for early stopping of training       |
| batch           | int                     | 16            | Number of images per batch (-1 for AutoBatch)                                     |
| imgsz           | int                     | 640           | Size of input images as integer                                                   |
| save            | bool                    | true          | Save train checkpoints and predict results                                        |
| save_period     | int                     | -1            | Save checkpoint every x epochs (disabled if < 1)                                  |
| cache           | str \| bool             | false         | true/ram, disk or false. Use cache for data loading                               |
| device          | int \| list[int] \| str | null          | Device to run on, e.g. cuda device=0 or device=0,1,2,3 or device=cpu              |
| workers         | int                     | 8             | Number of worker threads for data loading (per RANK if DDP)                       |
| project         | str                     | null          | Project name (path to the output directory)                                       |
| name            | str                     | null          | Experiment name                                                                   |
| exist_ok        | bool                    | false         | Whether to overwrite existing experiment                                          |
| pretrained      | str \| bool             | true          | Whether to use a pretrained model (bool) or a model to load weights from (str)    |
| optimizer       | str                     | 'auto'        | Optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto] |
| verbose         | bool                    | false         | Whether to print verbose output                                                   |
| seed            | int                     | 0             | Random seed for reproducibility                                                   |
| deterministic   | bool                    | true          | Whether to enable deterministic mode                                              |
| single_cls      | bool                    | false         | Train multi-class data as single-class                                            |
| rect            | bool                    | false         | Rectangular training with each batch collated for minimum padding                 |
| cos_lr          | bool                    | false         | Use cosine learning rate scheduler                                                |
| close_mosaic    | int                     | 10            | Disable mosaic augmentation for final epochs (0 to disable)                       |
| resume          | bool                    | false         | resume training from last checkpoint                                              |
| amp             | bool                    | true          | Automatic Mixed Precision (AMP) training, choices=[True, False]                   |
| fraction        | float                   | 1.0           | Dataset fraction to train on (default is 1.0, all images in train set)            |
| profile         | bool                    | false         | Profile ONNX and TensorRT speeds during training for loggers                      |
| freeze          | int \| list[int]        | null          | Freeze first n layers, or freeze list of layer indices during training            |
| lr0             | float                   | 0.01          | Initial learning rate (e.g. SGD=1E-2, Adam=1E-3)                                  |
| lrf             | float                   | 0.01          | Final learning rate (lr0 * lrf)                                                   |
| momentum        | float                   | 0.937         | SGD momentum/Adam beta1                                                           |
| weight_decay    | float                   | 0.0005        | Optimizer weight decay 5e-4                                                       |
| warmup_epochs   | float                   | 3.0           | Warmup epochs (fractions ok)                                                      |
| warmup_momentum | float                   | 0.8           | Warmup initial momentum                                                           |
| warmup_bias_lr  | float                   | 0.1           | Warmup initial bias lr                                                            |
| box             | float                   | 7.5           | Box loss gain                                                                     |
| cls             | float                   | 0.5           | Cls loss gain (scale with pixels)                                                 |
| dfl             | float                   | 1.5           | Dfl loss gain                                                                     |
| pose            | float                   | 12.0          | Pose loss gain (pose-only)                                                        |
| kobj            | float                   | 2.0           | Keypoint obj loss gain (pose-only)                                                |
| label_smoothing | float                   | 0.0           | Label smoothing (fraction)                                                        |
| nbs             | int                     | 64            | Nominal batch size                                                                |
| overlap_mask    | bool                    | true          | Masks should overlap during training (segment train only)                         |
| mask_ratio      | int                     | 4             | Mask downsample ratio (segment train only)                                        |
| dropout         | float                   | 0.0           | Use dropout regularization (classify train only)                                  |
| val             | bool                    | true          | Validate/test during training                                                     |

## Validation

The list of possible parameters for this section:

| Parameter name | Type                    | Default value | Explanation                                                        |
|----------------|-------------------------|---------------|--------------------------------------------------------------------|
| data           | str                     | null          | Path to data file, e.g. coco128.yaml                               |
| imgsz          | int                     | 640           | Size of input images as integer                                    |
| batch          | int                     | 16            | Number of images per batch (-1 for AutoBatch)                      |
| save_json      | bool                    | false         | Save results to JSON file                                          |
| save_hybrid    | bool                    | false         | Save hybrid version of labels (labels + additional predictions)    |
| conf           | float                   | 0.001         | Object confidence threshold for detection                          |
| iou            | float                   | 0.6           | Intersection over union (IoU) threshold for NMS                    |
| max_det        | int                     | 300           | Maximum number of detections per image                             |
| half           | bool                    | true          | Use half precision (FP16)                                          |
| device         | int \| list[int] \| str | null          | Device to run on, e.g. cuda device=0/1/2/3 or device=cpu           |
| dnn            | bool                    | false         | Use OpenCV DNN for ONNX inference                                  |
| plots          | bool                    | false         | Show plots during training                                         |
| rect           | bool                    | false         | Rectangular val with each batch collated for minimum padding       |
| split          | str                     | 'val'         | Dataset split to use for validation, e.g. 'val', 'test' or 'train' |
