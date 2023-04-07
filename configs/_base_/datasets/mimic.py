# dataset settings
dataset_type = 'MIMIC'
# dataset_type = 'ImageNet'
classes = [str(i) for i in range(14)]

img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Rotate', angle=15., prob=1),
    dict(type='Resize', size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=32, # batch size
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_prefix='/shared/dqwang/scratch/tongchen/MIMIC/train',
        ann_file='/shared/dqwang/scratch/yunkunzhang/mimic_multi-label_ann/train.txt',
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='/shared/dqwang/scratch/tongchen/MIMIC/test',
        ann_file='/shared/dqwang/scratch/yunkunzhang/mimic_multi-label_ann/test.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix='/shared/dqwang/scratch/tongchen/MIMIC/test',
        ann_file='/shared/dqwang/scratch/yunkunzhang/mimic_multi-label_ann/test.txt',
        classes=classes,
        pipeline=test_pipeline,
        test_mode=True))
evaluation = dict(interval=1, metric=['bag_class_auc_micro', 'bag_class_auc_macro','bag_class_auc_weighted'])