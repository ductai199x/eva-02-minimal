import torch
from model.proposal_generator import StandardRPNHead, Matcher
from model.anchor import DefaultAnchorGenerator
from model.box import Box2BoxTransform
from model.roi_heads import FastRCNNConvFCHead, FastRCNNOutputLayers, MaskRCNNConvUpsampleHead, ROIPooler
from model.backbone.conv_layers import LayerNormOverChannels


img_size = 1024
patch_size = 16
window_size = 16
embed_dim = 768
depth = 12


def get_model_config(train_config):
    num_classes = train_config["num_classes"]
    n_dets_per_image = train_config["n_dets_per_image"]
    det_score_threshold = train_config["det_score_threshold"]
    use_fed_loss = train_config["use_fed_loss"]
    get_fed_loss_cls_weights = train_config["get_fed_loss_cls_weights"]

    return {
        "backbone_config": {
            "vit_config": {
                "img_size": img_size,
                "patch_size": patch_size,
                "window_size": window_size,
                "embed_dim": embed_dim,
                "depth": depth,
                "num_heads": 12,
                "mlp_ratio": 4 * 2 / 3,
                "drop_path_rate": 0.1,
                "qkv_bias": True,
                "window_block_idxs": [i for i in range(depth) if (i + 1) % 3],
                "act_layer": torch.nn.SiLU,
            },
            "fpn_config": {
                "in_chans": embed_dim,
                "out_chans": 256,
                "patch_size": patch_size,
                "scale_factors": (4.0, 2.0, 1.0, 0.5),
                "top_block": True,
            },
        },
        "proposal_generator_config": {
            "in_features": ["p2", "p3", "p4", "p5", "p6"],
            "head": StandardRPNHead(
                in_channels=256,
                num_anchors=3,
                conv_dims=[-1, -1],
            ),
            "anchor_generator": DefaultAnchorGenerator(
                sizes=[[32], [64], [128], [256], [512]],
                aspect_ratios=[0.5, 1.0, 2.0],
                strides=[4, 8, 16, 32, 64],
                offset=0.0,
            ),
            "anchor_matcher": Matcher(
                thresholds=[0.3, 0.7], labels=[0, -1, 1], allow_low_quality_matches=True
            ),
            "box2box_transform": Box2BoxTransform(weights=[1.0, 1.0, 1.0, 1.0]),
            "batch_size_per_image": 256,
            "positive_fraction": 0.5,
            "pre_nms_topk": (2000, 1000),
            "post_nms_topk": (1000, 1000),
            "nms_thresh": 0.7,
        },
        "roi_heads_config": {
            "num_classes": num_classes,
            "batch_size_per_image": 512,
            "positive_fraction": 0.25,
            "box_in_features": ["p2", "p3", "p4", "p5"],
            "box_pooler": ROIPooler(
                output_size=7,
                scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
                sampling_ratio=0,
                pooler_type="ROIAlignV2",
            ),
            "mask_in_features": ["p2", "p3", "p4", "p5"],
            "mask_pooler": ROIPooler(
                output_size=14,
                scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
                sampling_ratio=0,
                pooler_type="ROIAlignV2",
            ),
            "mask_head": MaskRCNNConvUpsampleHead(
                input_shape={"channels": 256, "height": 14, "width": 14, "stride": None},
                num_classes=num_classes,
                conv_dims=[256, 256, 256, 256, 256],
                conv_norm=LayerNormOverChannels,
            ),
            "box_heads": [
                FastRCNNConvFCHead(
                    input_shape={"channels": 256, "height": 7, "width": 7, "stride": None},
                    conv_dims=[256, 256, 256, 256],
                    fc_dims=[1024],
                    conv_norm=LayerNormOverChannels,
                )
                for _ in range(3)
            ],
            "box_predictors": [
                FastRCNNOutputLayers(
                    input_shape={"channels": 1024},
                    box2box_transform=Box2BoxTransform(weights=(w1, w1, w2, w2)),
                    num_classes=num_classes,
                    test_score_thresh=det_score_threshold,
                    test_topk_per_image=n_dets_per_image,
                    cls_agnostic_bbox_reg=True,
                    use_sigmoid_ce=True,
                    use_fed_loss=use_fed_loss,
                    get_fed_loss_cls_weights=get_fed_loss_cls_weights,
                )
                for (w1, w2) in [(10, 5), (20, 10), (30, 15)]
            ],
            "proposal_matchers": [
                Matcher(thresholds=[th], labels=[0, 1], allow_low_quality_matches=False)
                for th in [0.5, 0.6, 0.7]
            ],
        },
    }
