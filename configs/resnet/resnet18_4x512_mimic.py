_base_ = [
    '../_base_/models/resnet18_mimic.py', '../_base_/datasets/mimic.py',
    '../_base_/schedules/imagenet_bs1024_coslr.py', '../_base_/default_runtime.py'
]

optimizer = dict(type='SGD', lr=1e-5, momentum=0.9, weight_decay=1e-4)
runner = dict(type='EpochBasedRunner', max_epochs=50)

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
        # dict(
        #     type='MMClsWandbHook',
        #     init_kwargs=dict(
        #         project='Distillaiton_MIMIC_MMCLS',
        #         entity='tongchen',
        #         name='train'
        #     ),
        #     log_checkpoint=True,
        #     log_checkpoint_metadata=True,
        # )
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='Distillaiton_MIMIC_MMCLS',
                entity='tongchen',
                name='train'
            ),
        )
    ])

load_from = '/shared/dqwang/scratch/lfzhou/r18_imgpre.pth'