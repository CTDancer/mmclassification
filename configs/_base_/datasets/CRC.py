# dataset settings
dataset_type = 'CRC'
classes = ['MSIMUT', 'MSS']

img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=64, # batch size
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_prefix='CRC/CRC_DX_train',
        ann_file='CRC/annotation/train_ann.txt',
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='CRC/CRC_DX_test',
        ann_file='CRC/annotation/test_ann.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix='CRC/CRC_DX_test',
        ann_file='CRC/annotation/test_ann.txt',
        classes=classes,
        pipeline=test_pipeline,
        test_mode=True))
evaluation = dict(interval=1, metric='auc')