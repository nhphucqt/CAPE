log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=20)
evaluation = dict(
    interval=25,
    metric=['PCK', 'NME', 'AUC', 'EPE'],
    key_indicator='PCK',
    gpu_collect=True,
    res_folder='')
optimizer = dict(type='Adam', lr=1e-05)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[160, 180])
total_epochs = 100
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
channel_cfg = dict(
    num_output_channels=1,
    dataset_joints=1,
    dataset_channel=[[0]],
    inference_channel=[0],
    max_kpt_num=100)
model = dict(
    type='TransformerPoseTwoStage',
    pretrained='torchvision://resnet50',
    encoder_config=dict(type='ResNet', depth=50, out_indices=(3, )),
    keypoint_head=dict(
        type='CapeFormerTwoStageHead',
        in_channels=2048,
        transformer=dict(
            type='CapeFormerTwoStageSupportRefineTransformer',
            d_model=256,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=2048,
            dropout=0.1,
            similarity_proj_dim=256,
            dynamic_proj_dim=128,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=True),
        share_kpt_branch=False,
        num_decoder_layer=3,
        with_heatmap_loss=True,
        heatmap_loss_weight=2.0,
        support_embedding_type="fixed",
        num_support=100,
        support_order_dropout=-1,
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True)),
    share_backbone=True,
    # training and testing settings
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=False,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))
data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=1,
    num_joints=1,
    dataset_channel=[[0]],
    inference_channel=[0])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=15,
        scale_factor=0.15),
    dict(type='TopDownAffineFewShot'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTargetFewShot', sigma=1),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs', 'category_id', 'skeleton'
        ])
]
valid_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffineFewShot'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTargetFewShot', sigma=1),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs', 'category_id', 'skeleton'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffineFewShot'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTargetFewShot', sigma=1),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs', 'category_id', 'skeleton'
        ])
]
# data_root = f'{data_root}'
data_root = '/volume/datasets/mp100'
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type='TransformerPoseDataset',
        ann_file=f'{data_root}/annotations/mp100_split1_train.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=dict(
            image_size=[256, 256],
            heatmap_size=[64, 64],
            num_output_channels=1,
            num_joints=1,
            dataset_channel=[[0]],
            inference_channel=[0]),
        valid_class_ids=None,
        max_kpt_num=100,
        num_shots=1,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='TopDownGetRandomScaleRotation',
                rot_factor=15,
                scale_factor=0.15),
            dict(type='TopDownAffineFewShot'),
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            dict(type='TopDownGenerateTargetFewShot', sigma=1),
            dict(
                type='Collect',
                keys=['img', 'target', 'target_weight'],
                meta_keys=[
                    'image_file', 'joints_3d', 'joints_3d_visible', 'center',
                    'scale', 'rotation', 'bbox_score', 'flip_pairs',
                    'category_id', 'skeleton'
                ])
        ]),
    val=dict(
        type='TransformerPoseDataset',
        ann_file=f'{data_root}/annotations/mp100_split1_val.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=dict(
            image_size=[256, 256],
            heatmap_size=[64, 64],
            num_output_channels=1,
            num_joints=1,
            dataset_channel=[[0]],
            inference_channel=[0]),
        valid_class_ids=None,
        max_kpt_num=100,
        num_shots=1,
        num_queries=15,
        num_episodes=100,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='TopDownAffineFewShot'),
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            dict(type='TopDownGenerateTargetFewShot', sigma=1),
            dict(
                type='Collect',
                keys=['img', 'target', 'target_weight'],
                meta_keys=[
                    'image_file', 'joints_3d', 'joints_3d_visible', 'center',
                    'scale', 'rotation', 'bbox_score', 'flip_pairs',
                    'category_id', 'skeleton'
                ])
        ]),
    test=dict(
        type='TestPoseVoronoiDataset',
        ann_file=f'{data_root}/annotations/human_face_split.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=dict(
            image_size=[256, 256],
            heatmap_size=[64, 64],
            num_output_channels=1,
            num_joints=1,
            dataset_channel=[[0]],
            inference_channel=[0]),
        valid_class_ids=None,
        max_kpt_num=100,
        num_shots=1,
        keep_orig=True,
        num_sample_kpts=20,
        num_queries=1,
        num_episodes=300000,

        pck_threshold_list=[0.05, 0.1, 0.15, 0.2, 0.25],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='TopDownAffineFewShot'),
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            dict(type='TopDownGenerateTargetFewShot', sigma=1),
            dict(
                type='Collect',
                keys=['img', 'target', 'target_weight', 'target_coarse'],
                meta_keys=[
                    'image_file', 'joints_3d', 'joints_3d_visible', 'center',
                    'scale', 'rotation', 'bbox_score', 'flip_pairs',
                    'category_id', 'skeleton'
                ])
        ]))
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ],
    name='visualizer')
shuffle_cfg = dict(interval=1)
