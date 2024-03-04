# dataset settings
dataset_type = 'CIFAR10'
img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=True)
transforms = [
    dict(type='ShiftScaleRotate', p=0.5),
    dict(type='RGBShift', p=0.1),
    dict(type='ChannelShuffle', p=0.1),
    dict(type='ColorJitter', p=0.1),
    dict(type='OneOf', p=0.1, 
         transforms=[dict(type='Blur', blur_limit=3, p=1.0),
                     dict(type='MedianBlur', blur_limit=3, p=1.0)]),
]
train_pipeline = [
    dict(type='Albu', transforms=transforms),
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type, data_prefix='data/cifar10',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type, data_prefix='data/cifar10', pipeline=test_pipeline),
    test=dict(
        type=dataset_type, data_prefix='data/cifar10', pipeline=test_pipeline))
