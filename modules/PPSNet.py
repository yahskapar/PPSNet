# Our proposed PPSNet, which includes a depth estimation backbone and a depth refinement module
# Part of this code is inspired by the Depth Anything (CVPR 2024) paper's code implementation
# See https://github.com/LiheYoung/Depth-Anything/blob/main/depth_anything/dpt.py

import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

from modules.blocks import FeatureFusionBlock, _make_scratch
from losses.calculate_PPL import calculate_per_pixel_lighting
from modules.unet import UNet

import numpy as np
import matplotlib.colors as mcolors
import utils.optical_flow_funs as OF

def _make_fusion_block(features, use_bn, size = None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )

def _rgb_to_grayscale(input_tensor):
    # Assuming input_tensor is of shape (B, C, H, W) and C=3 for RGB
    return torch.matmul(input_tensor.permute(0, 2, 3, 1), torch.tensor([0.2989, 0.5870, 0.1140], device=input_tensor.device)).unsqueeze(1)

def _normalize_vectors(v):
    """ Normalize a batch of 3D vectors to unit vectors using PyTorch. """
    norms = v.norm(p=2, dim=1, keepdim=True)
    return v / (norms + 1e-8)  # Adding a small epsilon to avoid division by zero

def _image_derivatives(image,diff_type='center'):
    c = image.size(1)
    if diff_type=='center':
        sobel_x = 0.5*torch.tensor([[0.0,0,0],[-1,0,1],[0,0,0]],device=image.device).view(1,1,3,3).repeat(c,1,1,1)
        sobel_y = 0.5*torch.tensor([[0.0,1,0],[0,0,0],[0,-1,0]],device=image.device).view(1,1,3,3).repeat(c,1,1,1)
    elif diff_type=='forward':
        sobel_x = torch.tensor([[0.0,0,0],[0,-1,1],[0,0,0]],device=image.device).view(1,1,3,3).repeat(c,1,1,1)
        sobel_y = torch.tensor([[0.0,1,0],[0,-1,0],[0,0,0]],device=image.device).view(1,1,3,3).repeat(c,1,1,1)
    
    dp_du = torch.nn.functional.conv2d(image,sobel_x,padding=1,groups=3)
    dp_dv = torch.nn.functional.conv2d(image,sobel_y,padding=1,groups=3)
    return dp_du, dp_dv

def _point_cloud_to_normals(pc, diff_type='center'):
    #pc (b,3,m,n)
    #return (b,3,m,n)
    dp_du, dp_dv = _image_derivatives(pc,diff_type=diff_type)
    normal = torch.nn.functional.normalize( torch.cross(dp_du,dp_dv,dim=1))
    return normal

def _get_normals_from_depth(depth, intrinsics, depth_is_along_ray=False , diff_type='center', normalized_intrinsics=True):
    #depth (b,1,m,n)
    #intrinsics (b,3,3)
    #return (b,3,m,n), (b,3,m,n)
    dirs = OF.get_camera_pixel_directions(depth.shape[2:4], intrinsics, normalized_intrinsics=normalized_intrinsics)
    dirs = dirs.permute(0,3,1,2)
    if depth_is_along_ray:
        dirs = torch.nn.functional.normalize(dirs,dim=1)
    pc = dirs*depth
    
    normal = _point_cloud_to_normals(pc, diff_type=diff_type)
    return normal, pc

class CrossAttentionModule(nn.Module):
    def __init__(self, feature_dim=384, heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=heads, batch_first=True)  # Note the batch_first=True

    def forward(self, queries, keys_values):
        # Since we're ignoring class tokens, the input is directly [B, N, C]
        # Apply multi-head attention; assuming queries and keys_values are prepared [B, N, C]
        attn_output, _ = self.attention(queries, keys_values, keys_values)

        # attn_output is already in the correct shape [B, N, C], so we return it directly
        return attn_output

class FeatureEncoder(nn.Module):
    """Encodes combined features into a lower-dimensional representation."""
    def __init__(self, input_channels, encoded_dim):
        super(FeatureEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, input_channels // 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(input_channels // 2, encoded_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)

class FiLM(nn.Module):
    """Applies Feature-wise Linear Modulation to condition disparity refinement."""
    def __init__(self, encoded_dim, target_channels):
        super(FiLM, self).__init__()
        self.scale_shift_net = nn.Linear(encoded_dim, target_channels * 2)

    def forward(self, features, disparity):
        # Global average pooling and processing to get scale and shift parameters
        pooled_features = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
        scale_shift_params = self.scale_shift_net(pooled_features)
        scale, shift = scale_shift_params.chunk(2, dim=1)
        
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        
        # Apply FiLM modulation to disparity
        modulated_disparity = disparity * scale + shift
        return modulated_disparity

class PPSNet_Refinement(nn.Module):
    """Refines disparity map conditioned on encoded features."""

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def __init__(self, disparity_channels, encoded_dim):
        super(PPSNet_Refinement, self).__init__()
        self.encoder = FeatureEncoder(input_channels=384, encoded_dim=encoded_dim)
        self.film = FiLM(encoded_dim=encoded_dim, target_channels=disparity_channels)
        self.cross_attention = CrossAttentionModule()
        self.refinement_net = UNet(disparity_channels, disparity_channels)

        self.apply(self.init_weights)

    def forward(self, features_rgb, features_colored_dot_product, initial_disparity):

        # Feed in RGB features from before and colored dot product
        combined_features = []
        for features1, features2 in zip(features_rgb, features_colored_dot_product):
            # Extract feature maps
            feature_map1 = features1[0]  # Assuming the first element is the feature map, shape [B, N, C]
            feature_map2 = features2[0]  # Assuming the first element is the feature map, shape [B, N, C]
            dummy_cls_token = torch.zeros((feature_map1.shape[0], 1, feature_map1.shape[-1]), device=feature_map1.device)  # [B, 1, C]

            # Apply cross-attention directly
            attn_output = self.cross_attention(feature_map1, feature_map2)  # Assume this outputs shape [B, N, C]

            # Append both the attention output and a dummy class token to combined_features
            # This makes combined_features a list of tuples [(feature_map, class_token), ...]
            combined_features.append((attn_output, dummy_cls_token))

        # Reshape combined_features for processing
        combined_features_reshaped = combined_features[3][0].reshape(-1, 384, 37, 37)
        encoded_features = self.encoder(combined_features_reshaped)
        
        # Condition disparity refinement on encoded features
        modulated_disparity = self.film(encoded_features, initial_disparity)
        
        # Refine modulated disparity
        modulated_disparity = F.interpolate(modulated_disparity, scale_factor=0.5, mode='bilinear', align_corners=False)
        refined_disparity = self.refinement_net(modulated_disparity)
        refined_disparity = F.interpolate(refined_disparity, size=(518, 518), mode='bilinear', align_corners=False)
        return (refined_disparity + initial_disparity).squeeze(1)

class DPTHead(nn.Module):
    def __init__(self, nclass, in_channels, features=256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False):
        super(DPTHead, self).__init__()
        
        self.nclass = nclass
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32
        
        if nclass > 1:
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_1, nclass, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
            
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
                nn.ReLU(True),
                nn.Identity(),
            )
            
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        
        return out
        
class DPT_DINOv2(nn.Module):
    def __init__(self, encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024], use_bn=False, use_clstoken=False, localhub=True):
        super(DPT_DINOv2, self).__init__()
        
        assert encoder in ['vits', 'vitb', 'vitl']
        
        self.pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14'.format(encoder))

        dim = self.pretrained.blocks[0].attn.qkv.in_features
        
        self.depth_head = DPTHead(1, dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)

    def forward(self, x, ref_dirs, light_pos, light_dir, mu, n_intrinsics):

        h, w = x.shape[-2:]
        
        # Step 1: Initial depth prediction and normals from depth
        features_rgb = self.pretrained.get_intermediate_layers(x, 4, return_class_token=True)
        
        patch_h, patch_w = h // 14, w // 14

        disparity = self.depth_head(features_rgb, patch_h, patch_w)
        disparity = F.interpolate(disparity, size=(h, w), mode="bilinear", align_corners=True)
        disparity = F.relu(disparity)

        # Get depth from disparity
        depth = 1 / disparity
        depth = torch.clamp(depth, 0, 1)

        normal, _ = _get_normals_from_depth(depth, n_intrinsics)

        # Step 2: Get PPL info from initial depth
        pc_preds = depth.squeeze(1).unsqueeze(3)*ref_dirs
        l, a = calculate_per_pixel_lighting(pc_preds, light_pos, light_dir, mu)

        # Convert image to grayscale
        img_gray = _rgb_to_grayscale(x)  # Resulting shape: (B, 1, H, W)

        # Ensure L and A are in the same format as RGB (B, C, H, W)
        l = l.permute(0, 3, 1, 2)  # Rearrange l from [B, H, W, C] to [B, C, H, W]
        a = a.permute(0, 3, 1, 2)  # Rearrange a from [B, H, W, C] to [B, C, H, W]

        # Log transformation on A
        a = torch.log(a + 1e-8)

        # Min-Max normalization for A should ideally be based on dataset statistics,
        # Here we normalize based on the min and max of the current batch for simplicity
        a_min, a_max = a.min(), a.max()
        a = (a - a_min) / (a_max - a_min + 1e-8)

        # Define the threshold for specular highlights
        threshold = 0.98  # Adjust this threshold based on your data
        # Create a mask for pixels above the intensity threshold
        specular_mask = (img_gray > threshold).float()  # [B, 1, H, W]

        # Normalize l and normal
        l_norm = _normalize_vectors(l)
        normal_norm = _normalize_vectors(normal)

        # Compute dot product and apply attenuation
        dot_product = torch.sum(l_norm * normal_norm, dim=1, keepdim=True)  # Shape: (B, 1, H, W)
        dot_product_clamped = torch.clamp(dot_product, -1, 1)
        dot_product_attenuated = dot_product_clamped * a

        rgb_vis = x[0].permute(1, 2, 0).cpu().numpy()

        # Convert RGB to HSV
        hsv_vis = mcolors.rgb_to_hsv(rgb_vis)

        # Retain H and S, but set V to 1.0 (max brightness)
        hsv_albedo = np.copy(hsv_vis)
        hsv_albedo[:, :, 2] = 1.0  # Set V to 100% brightness

        # Convert back to RGB for visualization
        rgb_albedo_vis = mcolors.hsv_to_rgb(hsv_albedo)

        h, w = x.shape[-2:]

        img_gray = img_gray.repeat(1, 3, 1, 1)  # New shape will be [B, 3, H, W]
        dot_product_attenuated = dot_product_attenuated.repeat(1, 3, 1, 1)  # New shape will be [B, 3, H, W]
        
        # Assuming 'albedo_np' is your NumPy array containing the albedo image
        albedo_tensor = torch.from_numpy(rgb_albedo_vis).float()
        albedo_tensor = albedo_tensor.to('cuda').div(255.0) if albedo_tensor.max() > 1.0 else albedo_tensor.to('cuda')

        colored_dot_product_attenuated = albedo_tensor.permute(2, 0, 1).unsqueeze(0) * dot_product_attenuated

        # Feature extraction for dot_product_attenuated
        features_colored_dot_product = self.pretrained.get_intermediate_layers(colored_dot_product_attenuated, 4, return_class_token=True)

        return disparity, features_rgb, features_colored_dot_product

class PPSNet_Backbone(DPT_DINOv2, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__(**config)
