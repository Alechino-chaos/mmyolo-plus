_base_ = './yolov10-h_l_syncbn_fast_3xb4-accum3-80e_dota10-640.py'

deepen_factor = 0.33
widen_factor = 0.50
max_channels = 1024
backbone_channels = [128, 256, 512]
neck_channels = [128, 256, 512]
work_dir = 'work_dirs/dota10_yolov10-h_s_640_80e_rand_b4x3_accum3_fp32'

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

