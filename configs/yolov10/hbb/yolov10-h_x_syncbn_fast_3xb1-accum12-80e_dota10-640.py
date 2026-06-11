_base_ = './yolov10-h_x_syncbn_fast_3xb4-accum3-80e_dota10-640.py'

train_batch_size_per_gpu = 1
accumulative_counts = 12
work_dir = 'work_dirs/dota10_yolov10-h_x_640_80e_rand_b1x3_accum12_fp32'

train_dataloader = dict(batch_size=train_batch_size_per_gpu)
optim_wrapper = dict(accumulative_counts=accumulative_counts)

