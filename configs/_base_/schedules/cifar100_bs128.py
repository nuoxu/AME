# optimizer
# optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='SGD', lr=0.1, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[400, 700, 900])
runner = dict(type='EpochBasedRunner', max_epochs=1000)
