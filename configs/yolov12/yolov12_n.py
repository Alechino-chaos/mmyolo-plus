_base_ = ['./yolov12_n_syncbn_fast_8xb16-500e_coco.py']

data_root = '../datasets/dota_yolo_split_1024/' 

train_ann_file = 'train/annotations/train.json'
train_data_prefix = 'train/images/'
val_ann_file = 'val/annotations/val.json'
val_data_prefix = 'val/images/'

num_classes = 16

metainfo = dict(
    classes=('plane', 'baseball-diamond', 'bridge', 'ground-track-field', 
             'small-vehicle', 'large-vehicle', 'ship', 'tennis-court', 
             'basketball-court', 'storage-tank', 'soccer-ball-field', 
             'roundabout', 'harbor', 'swimming-pool', 'helicopter',
             'container-crane') 
)

train_batch_size_per_gpu = 4 
train_num_workers = 2
val_batch_size_per_gpu = 1
val_num_workers = 2

max_epochs = 300
close_mosaic_epochs = 10

model = dict(
    bbox_head=dict(
        head_module=dict(num_classes=num_classes)),
    train_cfg=dict(
        assigner=dict(num_classes=num_classes)))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo, 
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix)))

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo, 
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix)))

test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=data_root + val_ann_file)
test_evaluator = val_evaluator