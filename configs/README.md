hsv_h	0.015	image HSV-Hue augmentation (fraction)
hsv_s	0.7	image HSV-Saturation augmentation (fraction)
hsv_v	0.4	image HSV-Value augmentation (fraction)
degrees	0.0	image rotation (+/- deg)
translate	0.1	image translation (+/- fraction)
scale	0.5	image scale (+/- gain)
shear	0.0	image shear (+/- deg)
perspective	0.0	image perspective (+/- fraction), range 0-0.001
flipud	0.0	image flip up-down (probability)
fliplr	0.5	image flip left-right (probability)
mosaic	1.0	image mosaic (probability)
mixup	0.0	image mixup (probability)
copy_paste	0.0	segment copy-paste (probability)



epochs	100	number of epochs to train for
patience	50	epochs to wait for no observable improvement for early stopping of training
batch	16	number of images per batch (-1 for AutoBatch)
imgsz	640	size of input images as integer
save	True	save train checkpoints and predict results
save_period	-1	Save checkpoint every x epochs (disabled if < 1)
cache	False	True/ram, disk or False. Use cache for data loading
device	None	device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
workers	8	number of worker threads for data loading (per RANK if DDP)
project	None	project name
name	None	experiment name
exist_ok	False	whether to overwrite existing experiment
pretrained	True	(bool or str) whether to use a pretrained model (bool) or a model to load weights from (str)
optimizer	'auto'	optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
verbose	False	whether to print verbose output
seed	0	random seed for reproducibility
deterministic	True	whether to enable deterministic mode
single_cls	False	train multi-class data as single-class
rect	False	rectangular training with each batch collated for minimum padding
cos_lr	False	use cosine learning rate scheduler
close_mosaic	10	(int) disable mosaic augmentation for final epochs (0 to disable)
resume	False	resume training from last checkpoint
amp	True	Automatic Mixed Precision (AMP) training, choices=[True, False]
fraction	1.0	dataset fraction to train on (default is 1.0, all images in train set)
profile	False	profile ONNX and TensorRT speeds during training for loggers
freeze	None	(int or list, optional) freeze first n layers, or freeze list of layer indices during training
lr0	0.01	initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
lrf	0.01	final learning rate (lr0 * lrf)
momentum	0.937	SGD momentum/Adam beta1
weight_decay	0.0005	optimizer weight decay 5e-4
warmup_epochs	3.0	warmup epochs (fractions ok)
warmup_momentum	0.8	warmup initial momentum
warmup_bias_lr	0.1	warmup initial bias lr
box	7.5	box loss gain
cls	0.5	cls loss gain (scale with pixels)
dfl	1.5	dfl loss gain
pose	12.0	pose loss gain (pose-only)
kobj	2.0	keypoint obj loss gain (pose-only)
label_smoothing	0.0	label smoothing (fraction)
nbs	64	nominal batch size
overlap_mask	True	masks should overlap during training (segment train only)
mask_ratio	4	mask downsample ratio (segment train only)
dropout	0.0	use dropout regularization (classify train only)
val	True	validate/test during training



data	None	path to data file, i.e. coco128.yaml
imgsz	640	size of input images as integer
batch	16	number of images per batch (-1 for AutoBatch)
save_json	False	save results to JSON file
save_hybrid	False	save hybrid version of labels (labels + additional predictions)
conf	0.001	object confidence threshold for detection
iou	0.6	intersection over union (IoU) threshold for NMS
max_det	300	maximum number of detections per image
half	True	use half precision (FP16)
device	None	device to run on, i.e. cuda device=0/1/2/3 or device=cpu
dnn	False	use OpenCV DNN for ONNX inference
plots	False	show plots during training
rect	False	rectangular val with each batch collated for minimum padding
split	val	dataset split to use for validation, i.e. 'val', 'test' or 'train'

