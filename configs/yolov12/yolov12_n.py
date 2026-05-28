_base_ = ['./yolov12_n_syncbn_fast_8xb16-500e_coco.py']

# ===============================
# Dataset
# ===============================

data_root = '../datasets/dota_yolo_split_1024/'

train_ann_file = 'train/annotations/train.json'
train_data_prefix = 'train/images/'

val_ann_file = 'val/annotations/val.json'
val_data_prefix = 'val/images/'

num_classes = 16

metainfo = dict(
    classes=(
        'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
        'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
        'basketball-court', 'storage-tank', 'soccer-ball-field',
        'roundabout', 'harbor', 'swimming-pool', 'helicopter',
        'container-crane'
    )
)

# ===============================
# Dataloader
# ===============================

train_batch_size_per_gpu = 4
train_num_workers = 2

val_batch_size_per_gpu = 1
val_num_workers = 2

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix)
    )
)

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix)
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=data_root + val_ann_file
)

test_evaluator = val_evaluator

# ===============================
# Model
# ===============================

model = dict(
    bbox_head=dict(
        head_module=dict(
            num_classes=num_classes
        )
    ),
    train_cfg=dict(
        assigner=dict(
            num_classes=num_classes
        )
    )
)

# ===============================
# Training schedule
# ===============================

max_epochs = 300
close_mosaic_epochs = 10

# 3 GPUs × 4 images/GPU = total batch size 12
# official config: 8 GPUs × 16 images/GPU = total batch size 128
# 0.01 × 12 / 128 = 0.0009375
base_lr = 0.0009375

optim_wrapper = dict(
    optimizer=dict(
        lr=base_lr,
        batch_size_per_gpu=train_batch_size_per_gpu
    )
)

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=10,
    dynamic_intervals=[(max_epochs - close_mosaic_epochs, 1)]
)

# 清空 base 里继承来的标准 param_scheduler，避免里面残留 end=500 / T_max=250
param_scheduler = []

# YOLOv12 实际主要使用这个 scheduler hook
default_hooks = dict(
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='cosine',
        lr_factor=0.01,
        max_epochs=max_epochs
    )
)

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49
    ),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - close_mosaic_epochs,
        switch_pipeline={{_base_.train_pipeline_stage2}}
    )
]