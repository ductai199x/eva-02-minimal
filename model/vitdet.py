import re

import torch
import torch.nn as nn
import torchvision.transforms.functional as tvF

from .backbone.vit import ViTDetBackBone
from .proposal_generator.rpn import RPN
from .roi_heads.cascade_rcnn import CascadeROIHeads
from .structures import ImageList
from .postprocessing import detector_postprocess as postprocess
from typing import *


class ViTDet(nn.Module):
    def __init__(
        self,
        backbone_config,
        proposal_generator_config,
        roi_heads_config,
    ):
        super().__init__()
        self.vit_embed_dim = backbone_config["vit_config"]["embed_dim"]
        self.backbone = ViTDetBackBone(**backbone_config)
        self.proposal_generator = RPN(**proposal_generator_config)
        self.roi_heads = CascadeROIHeads(**roi_heads_config)

        self.pixel_mean = nn.Parameter(
            torch.Tensor([123.6750, 116.2800, 103.5300]).view(3, 1, 1), requires_grad=False
        )
        self.pixel_std = nn.Parameter(
            torch.Tensor([58.3950, 57.1156, 57.3750]).view(3, 1, 1), requires_grad=False
        )

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        ckpt_state_dict = state_dict
        my_state_dict = self.state_dict()

        ckpt_keys = list(ckpt_state_dict.keys())
        my_keys = list(my_state_dict.keys())

        ckpt_to_my_mapping = {}
        outstanding_keys = list(my_keys)
        for ckpt_key in ckpt_keys:
            match_found = False
            ckpt_key_parts = set(ckpt_key.split("."))

            # Rule for attention weights and biases
            if (len(set(["q_proj", "k_proj", "v_proj", "q_bias", "k_bias", "v_bias"]).intersection(ckpt_key_parts)) > 0):  # fmt: skip
                if "weight" in ckpt_key:
                    block_idx, _, qkv, _ = ckpt_key.split(".")[-4::]
                    my_key = f"backbone.vit.blocks.{block_idx}.attn.qkv.weight"
                elif "bias" in ckpt_key:
                    block_idx, _, qkv = ckpt_key.split(".")[-3::]
                    my_key = f"backbone.vit.blocks.{block_idx}.attn.qkv.bias"

                if qkv[0] == "q":
                    my_state_dict[my_key][0 : self.vit_embed_dim] = ckpt_state_dict[ckpt_key]
                elif qkv[0] == "k":
                    my_state_dict[my_key][self.vit_embed_dim : self.vit_embed_dim * 2] = ckpt_state_dict[
                        ckpt_key
                    ]
                elif qkv[0] == "v":
                    my_state_dict[my_key][self.vit_embed_dim * 2 :] = ckpt_state_dict[ckpt_key]
                match_found = True
                continue

            # Rule for anchor_generator and fed_loss_cls_weights
            if "anchor_generator" in ckpt_key or "fed_loss_cls_weights" in ckpt_key:
                print("Skipping key:", ckpt_key)
                continue

            # Rule for everything else
            ckpt_key_matches = re.findall(r"(?=(?:^|\.)((?:\w+\.)*\w+)$)", ckpt_key)
            ckpt_key_matches = [
                m for m in ckpt_key_matches if len(re.findall(r"^\d*\.?(weight|bias)", m)) == 0
            ]

            for match in ckpt_key_matches:
                for key in outstanding_keys:
                    if key.find(match) != -1:
                        ckpt_to_my_mapping[ckpt_key] = key
                        outstanding_keys.remove(key)
                        match_found = True
                        break
                if match_found:
                    break
            assert (
                ckpt_state_dict[ckpt_key].size() == my_state_dict[key].size()
            ), f"Size mismatch for key '{ckpt_key}': {ckpt_state_dict[ckpt_key].size()} vs '{key}' {my_state_dict[key].size()}"
            my_state_dict[key] = ckpt_state_dict[ckpt_key]

            # Warn if no match found
            if not match_found:
                print(f"Warning! Key '{ckpt_key}' not found in model state_dict: {ckpt_key_matches}")

        super().load_state_dict(my_state_dict, strict=strict)

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = []
        original_sizes = []
        image_sizes = []
        for input in batched_inputs:
            image = input["image"].float().to(self.pixel_mean.device)
            image = (image - self.pixel_mean) / self.pixel_std
            orig_height, orig_width = image.shape[-2:]
            original_sizes.append((orig_height, orig_width))
            max_allowed_size = self.backbone.img_size

            if any([s > max_allowed_size for s in [orig_height, orig_width]]):
                if orig_height > orig_width:
                    new_height = max_allowed_size
                    new_width = int(orig_width / orig_height * max_allowed_size)
                else:
                    new_width = max_allowed_size
                    new_height = int(orig_height / orig_width * max_allowed_size)
                image = tvF.resize(image, (new_height, new_width))
                image = tvF.crop(image, 0, 0, max_allowed_size, max_allowed_size)
                image_sizes.append((new_height, new_width))
            else:
                image = tvF.crop(image, 0, 0, max_allowed_size, max_allowed_size)
                image_sizes.append((orig_height, orig_width))
            images.append(image)
        images = ImageList(torch.stack(images, dim=0), image_sizes)
        return images, original_sizes

    def forward(self, x: List[Dict[str, torch.Tensor]], final_mask_threshold=0.5):
        images, original_sizes = self.preprocess_image(x)
        features = self.backbone(images.tensor)

        proposals, proposal_losses = self.proposal_generator(images, features, None)
        proposals, detection_losses = self.roi_heads(features, proposals)
        if self.training:
            losses = {}
            losses.update(proposal_losses)
            losses.update(detection_losses)
            return losses
        else:
            results = []
            for i, instance in enumerate(proposals):
                H, W = original_sizes[i]
                result = postprocess(instance, H, W, final_mask_threshold)
                results.append(result)
            return results
