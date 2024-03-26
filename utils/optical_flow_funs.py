# Daniel Lichy
# FastNFPSCode
# Taken from https://github.com/dlichy/FastNFPSCode/blob/main/data_processing/optical_flow_funs.py

import torch

#pixel coordinate 0,0 is always located in the center of the top left pixel.
#depth_is_along_ray=True means depth is measured along the camera ray. depth_is_along_ray=False means depth is measure along the cameras z-axis.
#Think of an image as a functional  f:[-1,1]^2 -> R^Ch. And an image array of size (m,n) as the I[i,j] = f( (2j+1)/n - 1, (2i+1)/m - 1 ).
#So evaluating an image at normalized coordinates x,y is invariant (at least when upsampling by an odd factor) to resizing an image with cv2.resize(), which is the same as torch.nn.functional.interpolate(...,align_corners=False)

def mat_multiply(x,y):
	#x (b,i,j)
	#y (b,...,j)
	#return (b,...,i)
	s_y = y.shape
	#(b,-1,j)
	y = y.view(s_y[0],-1,s_y[-1])
	p = torch.einsum('bij,bkj->bki',x,y)
	p = p.view(*s_y[:-1],x.size(1))
	return p



def apply_affine(T,points):
	#T (b,k,k)
	#points (b,...,k-1)
	#return (b,k-1,...)
	s = points.shape
	points = points.view(s[0],-1,s[-1])
	k = T.size(1)-1
	R = T[:,0:k,0:k]
	t = T[:,0:k,k]
	trans_points = mat_multiply(R,points) + t.view(-1,1,k)
	trans_points = trans_points.view(s)
	return trans_points
	

def pixel_coords_to_directions(pts,intrinsics):
	#pts (b,...,2)
	#intrinsics (b,3,3)
	#return (b,...,3)
	z = torch.ones_like(pts[:,...,0:1])
	dirs_pix = torch.cat((pts,z),dim=-1)
	dirs = mat_multiply(torch.inverse(intrinsics),dirs_pix)
	return dirs
	


def get_pixel_to_normalized_coords_mat(image_shape,device='cpu'):
	M = torch.eye(3)
	M[0,0] = 2/image_shape[1]
	M[0,2] = 1/image_shape[1] - 1
	M[1,1] = 2/image_shape[0]
	M[1,2] = 1/image_shape[0] - 1
	return M.to(device)


def pixel_intrinsics_to_normalized_intrinsics(intrinsics, image_shape):
	M = get_pixel_to_normalized_coords_mat(image_shape)
	M = M.to(intrinsics.device).unsqueeze(0).repeat(intrinsics.size(0),1,1)
	intrinsics_n = torch.bmm(M,intrinsics)
	return intrinsics_n

def normalized_intrinsics_to_pixel_intrinsics(intrinsics_n, image_shape):
	M = get_pixel_to_normalized_coords_mat(image_shape)
	M_inv = torch.inverse(M).to(intrinsics_n.device).unsqueeze(0).repeat(intrinsics_n.size(0),1,1)
	intrinsics = torch.bmm(M_inv, intrinsics_n)
	return intrinsics
	

def get_coordinate_grid(image_shape,device='cpu'):
	#image_shape tuple = (m,n)
	#return (1,m,n,2)
	x_pix = torch.arange(image_shape[1], dtype=torch.float, device=device)
	y_pix = torch.arange(image_shape[0], dtype=torch.float, device=device)
	#(m,n)
	y,x = torch.meshgrid([y_pix,x_pix], indexing='ij') 
	pts = torch.stack((x.unsqueeze(0),y.unsqueeze(0)),dim=-1)
	return pts
	

def get_camera_pixel_directions( image_shape,intrinsics,normalized_intrinsics=True):
	# print(image_shape)
	# print(intrinsics.shape)
	#return (b,m,n,3)
	pts = get_coordinate_grid(image_shape,device=intrinsics.device)
	if normalized_intrinsics:
		pts = pixel_coords_to_normalized_coords(pts,image_shape)
	dirs = pixel_coords_to_directions(pts.repeat(intrinsics.size(0),1,1,1),intrinsics)
	return dirs

def pixel_coords_to_normalized_coords(pixel_coords,image_shape):
	#pixel_coords (b,...,2) 
	#return (b,...,2)
	M = get_pixel_to_normalized_coords_mat(image_shape)
	M = M.to(pixel_coords.device).unsqueeze(0).repeat(pixel_coords.size(0),1,1)
	return apply_affine(M,pixel_coords)
