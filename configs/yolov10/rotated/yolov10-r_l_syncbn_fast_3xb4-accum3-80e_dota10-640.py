_base_ = '../../_base_/default_runtime.py'

# ======================== Frequently modified parameters =====================
# ----- data related -----
data_root = 'data/DOTA1_yolo12x_hbb_split1024/'
train_ann_file = 'labels/train/'
train_data_prefix = 'images/train/'
val_ann_file = 'labels/val/'
val_data_prefix = 'images/val/'
submission_dir = './work_dirs/{{fileBasenameNoExtension}}/submission'

num_classes = 15
metainfo = dict(
    classes=(
        'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
        'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
        'basketball-court', 'storage-tank', 'soccer-ball-field',
        'roundabout', 'harbor', 'swimming-pool', 'helicopter'))

train_batch_size_per_gpu = 4
accumulative_counts = 3
train_num_workers = 8
persistent_workers = True

val_batch_size_per_gpu = 4
val_num_workers = 8
batch_shapes_cfg = None

# ----- train/val related -----
max_epochs = 80
warmup_epochs = 3
val_interval = 1
max_keep_ckpts = 3

# Base LR follows the existing DOTA-R AdamW setting, scaled from batch 8 to
# the effective batch 36: 0.00025 * 36 / 8 = 0.001125.
base_lr = 0.001125
lr_start_factor = 1.0e-5
weight_decay = 0.05
env_cfg = dict(cudnn_benchmark=True)

# ----- model related -----
img_scale = (640, 640)
dataset_type = 'YOLOv5YOLOTxtDataset'
random_rotate_ratio = 0.5
rotate_rect_obj_labels = [9, 11]

deepen_factor = 1.00
widen_factor = 1.00
max_channels = 512
backbone_channels = [256, 512, 512]
neck_channels = [256, 512, 512]
head_channels = 256
work_dir = 'work_dirs/dota10_yolov10-r_l_640_80e_rand_b4x3_accum3'

strides = [8, 16, 32]
angle_version = 'le90'
norm_cfg = dict(type='BN')
dsl_topk = 13
loss_cls_weight = 1.0
loss_bbox_weight = 2.0
qfl_beta = 2.0

model_test_cfg = dict(
    multi_label=True,
    decode_with_angle=True,
    nms_pre=30000,
    score_thr=0.05,
    nms=dict(type='nms_rotated', iou_threshold=0.1),
    max_per_img=2000)

# ============================== Unmodified in most cases =====================
model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False),
    backbone=dict(
        type='YOLOv10CSPDarknet',
        arch='P5',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        max_channels=max_channels,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        type='YOLOv10PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=1.0,
        max_channels=max_channels,
        in_channels=backbone_channels,
        out_channels=neck_channels,
        head_channels=head_channels,
        num_csp_blocks=3,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='RTMDetRotatedHead',
        head_module=dict(
            type='RTMDetRotatedSepBNHeadModule',
            num_classes=num_classes,
            widen_factor=1.0,
            in_channels=head_channels,
            stacked_convs=2,
            feat_channels=head_channels,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='SiLU', inplace=True),
            share_conv=True,
            pred_kernel_size=1,
            featmap_strides=strides),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0, strides=strides),
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True,
            beta=qfl_beta,
            loss_weight=loss_cls_weight),
        loss_bbox=dict(
            type='mmrotate.RotatedIoULoss',
            mode='linear',
            loss_weight=loss_bbox_weight),
        angle_version=angle_version,
        angle_coder=dict(type='mmrotate.PseudoAngleCoder'),
        use_hbbox_loss=False,
        loss_angle=None),
    train_cfg=dict(
        assigner=dict(
            type='BatchDynamicSoftLabelAssigner',
            num_classes=num_classes,
            topk=dsl_topk,
            iou_calculator=dict(type='mmrotate.RBboxOverlaps2D'),
            batch_iou=False),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=model_test_cfg)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True, box_type='hbox'),
    dict(
        type='mmrotate.ConvertBoxType',
        box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(
        type='mmrotate.RandomRotate',
        prob=random_rotate_ratio,
        angle_range=180,
        rotate_type='mmrotate.Rotate',
        rect_obj_labels=rotate_rect_obj_labels),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='RegularizeRotatedBox', angle_version=angle_version),
    dict(type='mmdet.PackDetInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True, box_type='hbox'),
    dict(
        type='mmrotate.ConvertBoxType',
        box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    collate_fn=dict(type='yolov5_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file=train_ann_file,
        data_prefix=dict(img_path=train_data_prefix),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file=val_ann_file,
        data_prefix=dict(img_path=val_data_prefix),
        test_mode=True,
        batch_shapes_cfg=batch_shapes_cfg,
        pipeline=val_pipeline))

val_evaluator = dict(type='mmrotate.DOTAMetric', metric='mAP')
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# Use epoch-based schedulers so OOM fallbacks with smaller micro-batches do not
# change warmup/cosine progress.
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=lr_start_factor,
        by_epoch=True,
        begin=0,
        end=warmup_epochs),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=warmup_epochs,
        end=max_epochs,
        T_max=max_epochs - warmup_epochs,
        by_epoch=True)
]

optim_wrapper = dict(
    type='OptimWrapper',
    accumulative_counts=accumulative_counts,
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=weight_decay),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=val_interval,
        max_keep_ckpts=max_keep_ckpts,
        save_best='auto'))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49)
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

visualizer = dict(type='mmrotate.RotLocalVisualizer')
load_from = None
