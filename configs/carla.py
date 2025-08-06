net = dict(
    type='RESANet',
)

backbone = dict(
    type='ResNetWrapper',
    resnet='resnet34',
    pretrained=True,
    replace_stride_with_dilation=[False, True, True],
    out_conv=True,
    fea_stride=8,
)

resa = dict(
    type='RESA',
    alpha=2.0,
    iter=4,
    input_channel=128,
    conv_stride=9,
)

decoder = 'BUSD'

trainer = dict(
    type='RESA'
)

evaluator = dict(
    type='BinaryIoUEvaluator',  # You will later implement this evaluator class
)

optimizer = dict(
    type='sgd',
    lr=0.02,
    weight_decay=1e-4,
    momentum=0.9,
)

epochs = 20
batch_size = 8
total_iter = (2000 // batch_size) * epochs  # adjust 2000 to actual training samples

import math
scheduler = dict(
    type='LambdaLR',
    lr_lambda=lambda _iter: math.pow(1 - _iter / total_iter, 0.9)
)

loss_type = 'bce_loss'
seg_loss_weight = 1.0
bg_weight = 1.0  # Not needed for BCE, but safe to leave in
ignore_label = 255

eval_ep = 1
save_ep = epochs
log_interval = 50


# Defaults for ImageNet normalization
# img_norm = dict(
#     mean=[0.485, 0.456, 0.406],  # ImageNet mean
#     std=[0.229, 0.224, 0.225]
# )
# Town04_2000 normalization:
img_norm = dict(
    mean = [0.5768, 0.5630, 0.5705],
    std = [0.1209, 0.1322, 0.1339]
)

img_height = 360
img_width = 640
cut_height = 0

dataset_path = './data/Carla'
dataset = dict(
    train=dict(
        type='CarlaLaneDataset',
        root=dataset_path,
        folder_name='Town04_2000',
        img_size=(img_height, img_width),
    ),
    val=dict(
        type='CarlaLaneDataset',
        root=dataset_path,
        folder_name='Town04_2000',
        img_size=(img_height, img_width),
    )
)

workers = 4
num_classes = 1  # binary mask + background, safe fallback
log_note = 'carla_binary_lane_seg'
