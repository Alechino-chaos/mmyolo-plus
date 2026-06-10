_base_ = './yolov10-r_l_syncbn_fast_3xb4-accum3-80e_dota10-640.py'

deepen_factor = 1.00
widen_factor = 1.25
max_channels = 512
backbone_channels = [320, 640, 640]
neck_channels = [320, 640, 640]
work_dir = 'work_dirs/dota10_yolov10-r_x_640_80e_rand_b4x3_accum3'

model = dict(
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        max_channels=max_channels),
    neck=dict(
        deepen_factor=deepen_factor,
        in_channels=backbone_channels,
        out_channels=neck_channels,
        max_channels=max_channels))
