import os
import cv2
import glob
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import PIL
from PIL import Image

# Imports for PPL calculation from Daniel Lichy's code
import utils.optical_flow_funs as OF

class C3VD_Dataset(Dataset):
    def __init__(self, data_dir, list, mode=None):
        self.data_dir = data_dir

        if mode in ('Train', 'Val'):
            with open(list, 'r') as f:
                self.dirs = f.readline().strip().split(' ')
        elif mode == 'Test':
            self.dirs = list
        else:
            raise ValueError("Mode not set or incorrect mode set! Only 'Train' or 'Test' is valid!")

        self.images = []
        self.depths = []
        self.normals = []
        self.intrinsic_matrices = []  # Store intrinsic matrices for each frame
        self.translation_vectors = []  # Store translation vectors for each frame

        for path in self.dirs:
            images = sorted(glob.glob(os.path.join(self.data_dir, path, 'images', '*.png')))
            depths = sorted(glob.glob(os.path.join(self.data_dir, path, 'depths', '*.tiff')))
            normals = sorted(glob.glob(os.path.join(self.data_dir, path, 'normals', '*.tiff')))

            # Load intrinsic parameters and translation vectors
            intrinsic_file = os.path.join(self.data_dir, path, 'camera.txt')
            camera_pose_file = os.path.join(self.data_dir, path, 'poses_gt.txt')

            intrinsic_matrix = self.load_intrinsic_parameters(intrinsic_file)
            T_vectors = self.load_translation_vectors(camera_pose_file)

            self.images.extend(images)
            self.depths.extend(depths)
            self.normals.extend(normals)
            self.intrinsic_matrices.extend([intrinsic_matrix] * len(images))
            self.translation_vectors.extend(T_vectors)

        image_size = 518
        self.trans_totensor = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR, antialias=False),
                                            transforms.CenterCrop(image_size),
                                            # transforms.ToTensor(),
                                            # transforms.Normalize(mean=0.5, std=0.5)
                                            ])
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)
    
    def load_intrinsic_parameters(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.read().splitlines()

        if len(lines) < 1:
            raise ValueError("Intrinsic parameters file is empty")

        # Split the first line (contains fx, fy, cx, cy)
        intrinsic_params = lines[0].split()

        # print(len(intrinsic_params))
        if len(intrinsic_params) != 6:
            raise ValueError("Invalid intrinsic parameters format")

        fx, fy, cx, cy = intrinsic_params[1:5]

        # Return the 3x3 intrinsic matrix as a numpy array
        intrinsic_matrix = np.array([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0, 0, 1]], dtype=np.float32)

        return intrinsic_matrix


    def load_translation_vectors(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.read().splitlines()

        T_vectors = []

        for line in lines:
            values = line.split()
            if len(values) != 17:
                raise ValueError("Invalid camera pose format")

            # Extract translation values (tx, ty, tz) from the last column of the matrix
            tx = float(values[4])
            ty = float(values[8])
            tz = float(values[12])

            # Append the translation vector as a list
            translation_vector = [tx, ty, tz]
            T_vectors.append(translation_vector)

        return T_vectors

    def __getitem__(self, idx):
        info = self.images[idx].split(os.path.sep)
        dataset, id = info[-3], info[-1].split('.')[0]

        image = cv2.imread(self.images[idx], cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(self.depths[idx], cv2.IMREAD_UNCHANGED)
        normal = cv2.imread(self.normals[idx], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        # Read image and normalize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.dtype == np.uint16:
            # Convert np.uint16 to np.uint8
            image = (image / 256).astype('uint8')
        image = image.astype(np.float32) / 255.0

        original_image_shape = image.shape[:2]

        # Read depth and scale appropriately
        depth = depth.astype(np.float32) / 65535.0

        # Scale normal appropriately
        normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
        normal = (normal / 32767.5) - 1.0
        normal = normal.astype(np.float32)

        min_depth = 0.0
        max_depth = 1.0

        # Create a mask to enable only depth values within the range
        mask = ((depth >= min_depth) & (depth <= max_depth)).astype(np.float32)

        # Retrieve the intrinsic matrix for the current frame
        intrinsic_matrix = self.intrinsic_matrices[idx]

        # Retrieve the translation vector for the current frame
        translation_vector = self.translation_vectors[idx]

        # Constants for Per-Pixel Lighting (PPL)
        pos = torch.zeros((3))              # light and camera co-located (b,3)
        direction = torch.tensor([0,0,1])   # light direction straight towards +z (b,3)
        mu = 0                              # approximate attenuation in air as 0 (b,)
        light_data = (pos, direction, mu)

        # Preprocessing
        image = self.transform(image)
        depth = self.transform(depth)
        normal = self.transform(normal)
        mask = self.transform(mask)

        image = self.trans_totensor(image)
        depth = self.trans_totensor(depth)
        normal = self.trans_totensor(normal)
        mask = self.trans_totensor(mask)

        # Update the intrinsics
        new_image_shape = image.shape[1:]
        # Adjust the matrix here
        original_height, original_width = original_image_shape
        new_height, new_width = new_image_shape
        # Compute scale factors
        scale_x = new_width / original_width
        scale_y = new_height / original_height

        # Update focal lengths
        new_fx = intrinsic_matrix[0, 0] * scale_x
        new_fy = intrinsic_matrix[1, 1] * scale_y

        new_cx = (intrinsic_matrix[0, 2] + 0.5) * scale_x - 0.5 
        new_cy = (intrinsic_matrix[1, 2] + 0.5) * scale_y - 0.5

        resized_intrinsic_matrix = np.array([[new_fx, 0, new_cx],
                                    [0, new_fy, new_cy],
                                    [0, 0, 1]], dtype=np.float32)

        # Normalized intrinsics (expected to be bx3x3)
        n_intrinsics = OF.pixel_intrinsics_to_normalized_intrinsics(torch.from_numpy(resized_intrinsic_matrix).unsqueeze(0).float(),new_image_shape).squeeze(0)

        return {
            'dataset': dataset,
            'id': id,
            'image': image,
            'depth': depth,
            'normal': normal,
            'mask': mask,
            'intrinsics': resized_intrinsic_matrix,
            'translation_vector': translation_vector,
            'light_data': light_data,
            'n_intrinsics': n_intrinsics
        }


if __name__ == '__main__':
    dataset = C3VD_Dataset(data_dir='/playpen-nas-ssd/akshay/3D_medical_vision/datasets/C3VD_registered_videos_undistorted_V3', 
                           list='/playpen-nas-ssd/akshay/3D_medical_vision/datasets/C3VD_registered_videos_undistorted_V3/val.txt', mode="Train")
    print(len(dataset))
