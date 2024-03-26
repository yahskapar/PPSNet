import torch
import torch.nn.functional as F
from torchvision import transforms

import PIL
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os.path
import sys
import re
import gc
import cv2
from tqdm import tqdm
from dataloaders.C3VD_dataloader import C3VD_Dataset
from scipy.optimize import leastsq

# Imports for Per-Pixel Lighting (PPL) calculation
import utils.optical_flow_funs as OF

# DepthAnything as backbone
from modules.original_backbone import DepthAnything

RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Create a general generator for use with the test dataloader
general_generator = torch.Generator()
general_generator.manual_seed(RANDOM_SEED)

# read_pfm() included alongside save_pfm() for reference
def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

def save_pfm(filename, image, scale=1):
    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()

def rel_percent_depth_difference_map(depth_gt, depth_est):
    """
    Compute a relative percent difference map between ground truth and estimated depth maps.

    depth_gt: Ground truth depth map (values range from 0 to max_depth).
    depth_est: Estimated depth map (values range from 0 to max_depth).

    Returns:
    errormap: Percent error map containing the percent difference between depth_gt and depth_est.
    """

    depth_gt = np.clip(depth_gt, a_min=1e-6, a_max=None)
    # Compute the percent difference between depth_gt and depth_est
    errormap = ((depth_gt - depth_est) / depth_gt) * 100.0

    return errormap

# Calculate scaling factor using least median squares method
def scale_predictions(gt, est):

    # Flatten the ground truth and estimated depth arrays
    gt_flat = gt.flatten()
    est_flat = est.flatten()

    # Calculate the scaling factor using least median squares
    def error_func(scale, gt, est):
        return np.median((gt - scale * est) ** 2)

    # Initial guess for the scaling factor
    initial_scale = 1.0

    # Use least median squares to find the optimal scaling factor
    result = leastsq(error_func, initial_scale, args=(gt_flat, est_flat))
    optimal_scale = result[0][0]

    # Scale the estimated depth array
    scaled_est = est * optimal_scale

    return scaled_est

parser = argparse.ArgumentParser(description='Evaluate Backbone')

parser.add_argument('--log_dir', type=str)
parser.add_argument('--ckpt', type=str)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--data_dir', type=str, default='/playpen-nas-ssd/akshay/3D_medical_vision/datasets/C3VD_registered_videos_undistorted_V3')
parser.add_argument('--test_list', type=str, default='./C3VD_splits/val.txt')
parser.add_argument('--num_workers', type=int, default=8)

args = parser.parse_args()

def test(testlist):
    # Initialize a list to store RMSE values for each sequence
    abs_rel_per_sequence = []
    sq_rel_per_sequence = []
    rmse_per_sequence = []
    rmse_log_per_sequence = []
    a1_per_sequence = []

    for scene in testlist:
        avg_abs_rel_sequence, avg_sq_rel_sequence, avg_rmse_sequence, \
        avg_rmse_log_sequence, avg_a1_sequence = test_sequence([scene])

        abs_rel_per_sequence.append(avg_abs_rel_sequence)
        sq_rel_per_sequence.append(avg_sq_rel_sequence)
        rmse_per_sequence.append(avg_rmse_sequence)
        rmse_log_per_sequence.append(avg_rmse_log_sequence)
        a1_per_sequence.append(avg_a1_sequence)

    # Print the RMSE values for each sequence and the overall RMSE
    for i, scene in enumerate(testlist):
        print(f"Sequence {scene}: RMSE (mm) = {rmse_per_sequence[i] * 1000.0}")

    # Calculate the overall RMSE by averaging the RMSE values across all sequences
    overall_abs_rel = np.mean(abs_rel_per_sequence)
    overall_sq_rel = np.mean(sq_rel_per_sequence)
    overall_rmse = np.mean(rmse_per_sequence)
    overall_rmse_log = np.mean(rmse_log_per_sequence)
    overall_a1 = np.mean(a1_per_sequence)
    print(f"Overall metrics across all sequences -> abs_rel: {overall_abs_rel}, sq_rel: {overall_sq_rel}, RMSE (mm): {overall_rmse * 1000.0}, RMSE_LOG: {overall_rmse_log}, a1: {overall_a1}")

def test_sequence(testlist):
    result_dir = os.path.join(args.log_dir, f'results')
    gt_dir = os.path.join(args.log_dir, 'gt')
    img_dir = os.path.join(args.log_dir, 'images')
    depth_errormap_dir = os.path.join(args.log_dir, 'depth_error_maps')
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(depth_errormap_dir, exist_ok=True)

    testSet = C3VD_Dataset(data_dir=args.data_dir, list=testlist, mode='Test')
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=1, 
                                            shuffle=False, num_workers=args.num_workers, generator=general_generator)

    image_size = 518
    preproc_trans = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                        transforms.CenterCrop(image_size),
                                        ])

    map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load DepthAnything
    model = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format('vits'))

    checkpoint = torch.load(args.ckpt, map_location=map_location)
    state_dict = {}
    for k, v in checkpoint['state_dict'].items():
        name = k[7:] if k.startswith('module.') else k
        state_dict[name] = v
    model.load_state_dict(state_dict)
    model.to(device)

    # Initialize a list to store RMSE values for each sequence
    abs_rel_sequence = []
    sq_rel_sequence = []
    rmse_sequence = []
    rmse_log_sequence = []
    a1_sequence = []

    with torch.no_grad():
        for batch in tqdm(testLoader):
            image = batch['image']
            img_tensor = preproc_trans(image)[:3].to(device)
            img_tensor = F.interpolate(img_tensor, size=(518, 518), mode='bicubic', align_corners=False)

            gt = batch['depth'].cuda()
            gt = preproc_trans(gt).float().squeeze(1)
            gt = F.interpolate(gt.unsqueeze(0), (518, 518), mode='bicubic').squeeze(0)
            gt = gt.clamp(0,1)

            ref_dirs = OF.get_camera_pixel_directions(img_tensor.shape[2:4], batch['n_intrinsics'], normalized_intrinsics=True).to(device)
            light_data = [item.to(device) for item in batch['light_data']]

            disp_pred = model(img_tensor) # DepthAnything returns disparity

            pred = 1 / disp_pred
            pred = torch.clamp(pred, 0, 1)

            # Normalize output from 0 to 1 after clamping outliers
            pred_max = pred.max()
            pred = pred / pred_max

            pred = F.interpolate(pred.unsqueeze(0), (384, 384), mode='bicubic').squeeze(0)
            pred = pred.clamp(0,1)  # Precaution?

            # Re-scale the img_tensor from 518x518 back to 384x384
            img_tensor = F.interpolate(img_tensor, size=(384, 384), mode='bicubic', align_corners=False)
            ref_dirs = OF.get_camera_pixel_directions(img_tensor.shape[2:4], batch['n_intrinsics'], normalized_intrinsics=True).to(device)
            pc_preds = pred.squeeze(1).unsqueeze(3)*ref_dirs

            gt = F.interpolate(gt.unsqueeze(0), (384, 384), mode='bicubic').squeeze(0)
            gt = gt.clamp(0,1)

            # Metrics calculation
            # Scale in meters for C3VD
            gt = (gt * 65535.0) * 0.000001525
            pred = (pred * 65535.0) * 0.000001525

            depth_est = pred.cpu().numpy()
            depth_gt = gt.cpu().numpy()

            depth_est = scale_predictions(depth_gt, depth_est)

            epsilon = 1e-6  # Small positive constant
            # Compute error metrics
            abs_rel = np.abs(depth_gt - depth_est) / (depth_gt + epsilon)
            sq_rel = np.mean(((depth_gt - depth_est) ** 2) / (depth_gt + epsilon))
            rmse = np.sqrt(np.mean((depth_gt - depth_est) ** 2))

            rmse_log = np.sqrt(np.mean(
                (np.log(depth_gt + epsilon) - np.log(depth_est + epsilon)) ** 2))
            a1 = np.mean(np.where(abs_rel < 0.1, 1.0, 0.0), dtype=np.float32)

            # Compute errormap
            errormap_depth = rel_percent_depth_difference_map(depth_gt, depth_est)
            errormap_depth = errormap_depth.squeeze()

            # Append per-frame metrics
            abs_rel_sequence.append(abs_rel)
            sq_rel_sequence.append(sq_rel)
            rmse_sequence.append(rmse)
            rmse_log_sequence.append(rmse_log)
            a1_sequence.append(a1)

            # Save predicted maps and GT maps as .pfm and .png files
            dataset_dir_img = os.path.join(img_dir, batch['dataset'][0])
            dataset_dir_pred = os.path.join(result_dir, batch['dataset'][0])
            dataset_dir_gt = os.path.join(gt_dir, batch['dataset'][0])
            dataset_dir_depth_errormap = os.path.join(depth_errormap_dir, batch['dataset'][0])

            if not os.path.exists(dataset_dir_pred):
                os.makedirs(dataset_dir_pred)
            if not os.path.exists(dataset_dir_gt):
                os.makedirs(dataset_dir_gt)
            if not os.path.exists(dataset_dir_img):
                os.makedirs(dataset_dir_img)
            if not os.path.exists(dataset_dir_depth_errormap):
                os.makedirs(dataset_dir_depth_errormap)

            # Save input image
            img_tensor = F.interpolate(img_tensor, (384, 384), mode='bicubic')
            img_tensor = img_tensor.clamp(0,1)
            img_tensor = img_tensor.cpu().numpy()
            img_vis = img_tensor.squeeze().transpose(1, 2, 0)
            # img_vis = (img_vis * 255).astype(np.uint8)
            plt.imsave(os.path.join(dataset_dir_img, f"{batch['id'][0]}.png"), img_vis)

            # Save PFM for depth
            save_pfm(os.path.join(dataset_dir_gt, f"{batch['id'][0]}.pfm"), depth_gt.squeeze())
            save_pfm(os.path.join(dataset_dir_pred, f"{batch['id'][0]}.pfm"), depth_est.squeeze())
            # Save PNG visualization for depth
            plt.imsave(os.path.join(dataset_dir_gt, f"{batch['id'][0]}.png"), depth_gt.squeeze(),cmap='jet')
            plt.imsave(os.path.join(dataset_dir_pred, f"{batch['id'][0]}.png"), depth_est.squeeze(),cmap='jet')

            # Save error map
            # Calculate the depth error as a percentage of the ground truth depth
            percent_errormap_depth = rel_percent_depth_difference_map(depth_gt, depth_est)

            cmap = plt.get_cmap('coolwarm')

            # Save the percent error map
            plt.imshow(percent_errormap_depth.squeeze(), cmap=cmap, vmin=-100, vmax=100)
            cbar = plt.colorbar()
            cbar.ax.set_ylabel('Percent Depth Error (%)')  # Update the label for the colorbar
            # Add custom labels above and below the colorbar with adjusted positioning
            cbar.ax.text(0.25, 1.05, 'Predicted Closer', transform=cbar.ax.transAxes, ha='left', va='center')
            cbar.ax.text(0.25, -0.05, 'Predicted Farther', transform=cbar.ax.transAxes, ha='left', va='center')

            # Save the percent error map
            plt.savefig(os.path.join(dataset_dir_depth_errormap, f"percent_depth_error_map_{batch['id'][0]}.png"))
            plt.close()

    # Calculate the average metrics for this sequence
    avg_abs_rel_sequence = np.mean(abs_rel_sequence)
    avg_sq_rel_sequence = np.mean(sq_rel_sequence)
    avg_rmse_sequence = np.mean(rmse_sequence)
    avg_rmse_log_sequence = np.mean(rmse_log_sequence)
    avg_a1_sequence = np.mean(a1_sequence)

    torch.cuda.empty_cache()
    gc.collect()
    return avg_abs_rel_sequence, avg_sq_rel_sequence, avg_rmse_sequence, avg_rmse_log_sequence, avg_a1_sequence

if __name__ == '__main__':
    with open(args.test_list) as f:
        content = f.readlines()
        testlist = [item for line in content for item in line.split()]
    test(testlist)
