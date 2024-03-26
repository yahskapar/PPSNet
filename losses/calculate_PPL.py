# Note: this code was taken from the Fast Light-Weight Near-Field Photometric Stereo (CVPR 2022) paper's code implementation. 
# See https://github.com/dlichy/FastNFPSCode/blob/c503d59ba8fcfc43ee8aa981a1ed375801b783a9/rendering/advanced_rendering_funs.py#L33-L47

import torch
import torch.nn.functional as F

def calculate_per_pixel_lighting(pc_preds, light_pos, light_dir, mu):
    #pc_gt and pc_preds (b,m,n,3)
    #light_pos (b,3)
    #light_dir (b,3)
    #angular attenuation mu (b,)
    #return (b,m,n,3) (b,m,n,1)

    # Calculate PPL for pc_preds
    to_light_vec_preds = light_pos.unsqueeze(1).unsqueeze(1) - pc_preds
    n_to_light_vec_preds = F.normalize(to_light_vec_preds,dim=3)
    #(b,m,n,1)
    len_to_light_vec_preds = torch.norm(to_light_vec_preds,dim=3,keepdim=True)
    light_dir_dot_to_light_preds = torch.sum(-n_to_light_vec_preds*light_dir.unsqueeze(1).unsqueeze(1),dim=3,keepdim=True).clamp(min=1e-8)
    numer_preds = torch.pow(light_dir_dot_to_light_preds, mu.view(-1,1,1,1))
    atten_preds = numer_preds/(len_to_light_vec_preds**2).clamp(min=1e-8)

    return n_to_light_vec_preds, atten_preds