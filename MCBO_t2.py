import argparse
import os
import time
import warnings
from pathlib import Path
from typing import Optional, Union

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch, sys, json
from torch.utils.data import Dataset
from scipy.ndimage import distance_transform_edt as edt

warnings.filterwarnings("ignore")

class SpatialTransformer(nn.Module):

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, src, flow, is_grid_out=False, mode=None, align_corners=True):

        new_locs = self.grid + flow
        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        if mode is None:
            out = F.grid_sample(src, new_locs, align_corners=align_corners, mode=self.mode)
        else:
            out = F.grid_sample(src, new_locs, align_corners=align_corners, mode=mode)

        if is_grid_out:
            return out, new_locs
        return out

class registerSTModel(nn.Module):

    def __init__(self, img_size=(64, 256, 256), mode='bilinear'):
        super(registerSTModel, self).__init__()

        self.spatial_trans = SpatialTransformer(img_size, mode)

    def forward(self, img, flow, is_grid_out=False, align_corners=True):
        out = self.spatial_trans(img, flow, is_grid_out=is_grid_out, align_corners=align_corners)

        return out

#enforce inverse consistency of forward and backward transform
def inverse_consistency(disp_field1s,disp_field2s,iter=20):
    B,C,H,W,D = disp_field1s.size()
    #make inverse consistent
    with torch.no_grad():
        disp_field1i = disp_field1s.clone()
        disp_field2i = disp_field2s.clone()

        identity = F.affine_grid(torch.eye(3,4).unsqueeze(0),(1,1,H,W,D)).permute(0,4,1,2,3).to(disp_field1s.device).to(disp_field1s.dtype)
        for i in range(iter):
            disp_field1s = disp_field1i.clone()
            disp_field2s = disp_field2i.clone()

            disp_field1i = 0.5*(disp_field1s-F.grid_sample(disp_field2s,(identity+disp_field1s).permute(0,2,3,4,1)))
            disp_field2i = 0.5*(disp_field2s-F.grid_sample(disp_field1s,(identity+disp_field2s).permute(0,2,3,4,1)))

    return disp_field1i,disp_field2i

#solve two coupled convex optimisation problems for efficient global regularisation
def coupled_convex_mean(ssd,ssd_argmin,disp_mesh_t,grid_sp,shape):
    H = int(shape[0]); W = int(shape[1]); D = int(shape[2]);

    disp_soft = F.avg_pool3d(disp_mesh_t.view(3,-1)[:,ssd_argmin.view(-1)].reshape(1,3,H//grid_sp,W//grid_sp,D//grid_sp),3,padding=1,stride=1)
    disp_soft_all = []
    coeffs = torch.tensor([0.003,0.01,0.03,0.1,0.3,1])
    for j in range(6):
        ssd_coupled_argmin = torch.zeros_like(ssd_argmin)
        with torch.no_grad():
            for i in range(H//grid_sp):

                coupled = ssd[:,i,:,:]+coeffs[j]*(disp_mesh_t-disp_soft[:,:,i].view(3,1,-1)).pow(2).sum(0).view(-1,W//grid_sp,D//grid_sp)
                ssd_coupled_argmin[i] = torch.argmin(coupled,0)
                # ssd_coupled_argmin[i] = torch.argmax(coupled, 0)

        disp_soft = F.avg_pool3d(disp_mesh_t.view(3,-1)[:,ssd_coupled_argmin.view(-1)].reshape(1,3,H//grid_sp,W//grid_sp,D//grid_sp),3,padding=1,stride=1)
        disp_soft_all.append(disp_soft)

    disp_soft_mean = sum(disp_soft_all) / len(disp_soft_all) ##mean

    return disp_soft_mean

#correlation layer: dense discretised displacements to compute SSD cost volume with box-filter
def correlate(mind_fix,mind_mov,disp_hw,grid_sp,shape, ch=12):
    H = int(shape[0]); W = int(shape[1]); D = int(shape[2]);

    with torch.no_grad():
        mind_unfold = F.unfold(F.pad(mind_mov,(disp_hw,disp_hw,disp_hw,disp_hw,disp_hw,disp_hw)).squeeze(0),disp_hw*2+1)
        mind_unfold = mind_unfold.view(ch,-1,(disp_hw*2+1)**2,W//grid_sp,D//grid_sp)

    ssd = torch.zeros((disp_hw*2+1)**3,H//grid_sp,W//grid_sp,D//grid_sp,dtype=mind_fix.dtype, device=mind_fix.device)#.cuda().half()
    with torch.no_grad():
        for i in range(disp_hw*2+1):
            mind_sum = (mind_fix.permute(1,2,0,3,4)-mind_unfold[:,i:i+H//grid_sp]).pow(2).sum(0,keepdim=True)
            ssd[i::(disp_hw*2+1)] = F.avg_pool3d(F.avg_pool3d(mind_sum.transpose(2,1),3,stride=1,padding=1),3,stride=1,padding=1).squeeze(1)
        ssd = ssd.view(disp_hw*2+1,disp_hw*2+1,disp_hw*2+1,H//grid_sp,W//grid_sp,D//grid_sp).transpose(1,0).reshape((disp_hw*2+1)**3,H//grid_sp,W//grid_sp,D//grid_sp)
        ssd_argmin = torch.argmin(ssd,0)#

    return ssd, ssd_argmin

def MINDSSC(img, radius=2, dilation=2):
    # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor

    # kernel size
    kernel_size = radius * 2 + 1

    # define start and end locations for self-similarity pattern
    six_neighbourhood = torch.Tensor([[0, 1, 1],
                                      [1, 1, 0],
                                      [1, 0, 1],
                                      [1, 1, 2],
                                      [2, 1, 1],
                                      [1, 2, 1]]).long()

    # squared distances
    dist = pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

    # define comparison mask
    x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
    mask = ((x > y).view(-1) & (dist == 2).view(-1))

    # build kernel
    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
    mshift1 = torch.zeros(12, 1, 3, 3, 3).cuda()
    mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
    mshift2 = torch.zeros(12, 1, 3, 3, 3).cuda()
    mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
    rpad1 = nn.ReplicationPad3d(dilation)
    rpad2 = nn.ReplicationPad3d(radius)

    # compute patch-ssd
    ssd = F.avg_pool3d(rpad2(
        (F.conv3d(rpad1(img), mshift1, dilation=dilation) - F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2),
                       kernel_size, stride=1)

    # MIND equation
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = torch.clamp(mind_var, mind_var.mean().item() * 0.001, mind_var.mean().item() * 1000)
    mind /= mind_var
    mind = torch.exp(-mind)

    # permute to have same ordering as C++ code
    mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]

    return mind

def pdist_squared(x):
    xx = (x**2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0, np.inf)
    return dist

def extract_features(
    img_fixed: torch.Tensor,
    img_moving: torch.Tensor,
    mind_r: int,
    mind_d: int,
    use_mask: bool,
    mask_fixed: torch.Tensor,
    mask_moving: torch.Tensor,
) -> (torch.Tensor, torch.Tensor):
    """Extract MIND and/or semantic nnUNet features"""

    # MIND features
    if use_mask:
        H,W,D = img_fixed.shape[-3:]

        #replicate masking
        avg3 = nn.Sequential(nn.ReplicationPad3d(1),nn.AvgPool3d(3,stride=1))
        avg3.cuda()
        
        mask = (avg3(mask_fixed.view(1,1,H,W,D).cuda())>0.9).float()
        _,idx = edt((mask[0,0,::2,::2,::2]==0).squeeze().cpu().numpy(),return_indices=True)
        fixed_r = F.interpolate((img_fixed[::2,::2,::2].cuda().reshape(-1)[idx[0]*D//2*W//2+idx[1]*D//2+idx[2]]).unsqueeze(0).unsqueeze(0),scale_factor=2,mode='trilinear')
        fixed_r.view(-1)[mask.view(-1)!=0] = img_fixed.cuda().reshape(-1)[mask.view(-1)!=0]

        mask = (avg3(mask_moving.view(1,1,H,W,D).cuda())>0.9).float()
        _,idx = edt((mask[0,0,::2,::2,::2]==0).squeeze().cpu().numpy(),return_indices=True)
        moving_r = F.interpolate((img_moving[::2,::2,::2].cuda().reshape(-1)[idx[0]*D//2*W//2+idx[1]*D//2+idx[2]]).unsqueeze(0).unsqueeze(0),scale_factor=2,mode='trilinear')
        moving_r.view(-1)[mask.view(-1)!=0] = img_moving.cuda().reshape(-1)[mask.view(-1)!=0]

        features_fix = MINDSSC(fixed_r.cuda(),mind_r,mind_d).half()
        features_mov = MINDSSC(moving_r.cuda(),mind_r,mind_d).half()
    else:
        img_fixed = img_fixed.unsqueeze(0).unsqueeze(0)
        img_moving = img_moving.unsqueeze(0).unsqueeze(0)
        features_fix = MINDSSC(img_fixed.cuda(),mind_r,mind_d).half()
        features_mov = MINDSSC(img_moving.cuda(),mind_r,mind_d).half()
    
    return features_fix, features_mov


def validate_image(img: Union[torch.Tensor, np.ndarray, sitk.Image], dtype=float) -> torch.Tensor:
    """Validate image input"""
    if not isinstance(img, torch.Tensor):
        if isinstance(img, sitk.Image):
            img = sitk.GetArrayFromImage(img)
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img.astype(dtype))
        else:
            raise ValueError("Input image must be a torch.Tensor, a numpy.ndarray or a SimpleITK.Image")
    return img

def convex_adam_pt(
    img_fixed: Union[torch.Tensor, np.ndarray, sitk.Image],
    img_moving: Union[torch.Tensor, np.ndarray, sitk.Image],
    mind_r: int = 2,
    mind_d: int = 2,
    lambda_weight: float = 1.25,
    grid_sp: int = 6,
    disp_hw: int = 4,
    selected_niter: int = 80,
    selected_smooth: int = 0,
    grid_sp_adam: int = 2,
    ic: bool = True,
    use_mask: bool = False,
    path_fixed_mask: Optional[Union[Path, str]] = None,
    path_moving_mask: Optional[Union[Path, str]] = None,
) -> None:
    """Coupled convex optimisation with adam instance optimisation"""
    img_fixed = validate_image(img_fixed)
    img_moving = validate_image(img_moving)
    img_fixed = img_fixed.float()
    img_moving = img_moving.float()

    if use_mask:
        mask_fixed = torch.from_numpy(nib.load(path_fixed_mask).get_fdata()).float()
        mask_moving = torch.from_numpy(nib.load(path_moving_mask).get_fdata()).float()
    else:
        mask_fixed = None
        mask_moving = None

    H, W, D = img_fixed.shape

    # compute features and downsample (using average pooling)
    with torch.no_grad():

        features_fix, features_mov = extract_features(img_fixed=img_fixed,
                                                      img_moving=img_moving,
                                                      mind_r=mind_r,
                                                      mind_d=mind_d,
                                                      use_mask=use_mask,
                                                      mask_fixed=mask_fixed,
                                                      mask_moving=mask_moving)

        features_fix_smooth = F.avg_pool3d(features_fix,grid_sp,stride=grid_sp)
        features_mov_smooth = F.avg_pool3d(features_mov,grid_sp,stride=grid_sp)

        n_ch = features_fix_smooth.shape[1]
    del features_fix, features_mov

    # compute correlation volume with SSD
    ssd,ssd_argmin = correlate(features_fix_smooth,features_mov_smooth,disp_hw,grid_sp,(H,W,D), n_ch)

    # provide auxiliary mesh grid
    disp_mesh_t = F.affine_grid(disp_hw*torch.eye(3,4).cuda().half().unsqueeze(0),(1,1,disp_hw*2+1,disp_hw*2+1,disp_hw*2+1),align_corners=True).permute(0,4,1,2,3).reshape(3,-1,1)

    # perform coupled convex optimisation
    disp_soft = coupled_convex_mean(ssd, ssd_argmin, disp_mesh_t, grid_sp, (H, W, D))

    del ssd, ssd_argmin
    #
    # if "ic" flag is set: make inverse consistent
    if ic:
        scale = torch.tensor([H//grid_sp-1,W//grid_sp-1,D//grid_sp-1]).view(1,3,1,1,1).cuda().half()/2

        ssd_,ssd_argmin_ = correlate(features_mov_smooth,features_fix_smooth,disp_hw,grid_sp,(H,W,D), n_ch)

        disp_soft_ = coupled_convex_mean(ssd_,ssd_argmin_,disp_mesh_t,grid_sp,(H,W,D))
        del ssd_, ssd_argmin_

        disp_ice,_ = inverse_consistency((disp_soft/scale).flip(1),(disp_soft_/scale).flip(1),iter=15)

        disp_hr = F.interpolate(disp_ice.flip(1)*scale*grid_sp,size=(H,W,D),mode='trilinear',align_corners=False)

    else:
        disp_hr=disp_soft

    x = disp_hr[0,0,:,:,:].cpu().half().data.numpy()
    y = disp_hr[0,1,:,:,:].cpu().half().data.numpy()
    z = disp_hr[0,2,:,:,:].cpu().half().data.numpy()
    displacements = np.stack((x,y,z),3).astype(float)
    torch.cuda.empty_cache()

    return displacements, disp_hr

def convex_adam_pyramid(
    img_fixed: Union[torch.Tensor, np.ndarray, sitk.Image],
    img_moving: Union[torch.Tensor, np.ndarray, sitk.Image],
    mind_r: int = 2,
    mind_d: int = 2,
    lambda_weight: float = 1.25,
    grid_sp: int = 6,
    disp_hw: int = 4,
    selected_niter: int = 80,
    selected_smooth: int = 0,
    grid_sp_adam: int = 2,
    ic: bool = True,
    use_mask: bool = False,
    path_fixed_mask: Optional[Union[Path, str]] = None,
    path_moving_mask: Optional[Union[Path, str]] = None,
) -> None:
    """Coupled convex optimisation with adam instance optimisation"""
    img_fixed = validate_image(img_fixed)
    img_moving = validate_image(img_moving)
    img_fixed = img_fixed.float()
    img_moving = img_moving.float()

    if use_mask:
        mask_fixed = torch.from_numpy(nib.load(path_fixed_mask).get_fdata()).float()
        mask_moving = torch.from_numpy(nib.load(path_moving_mask).get_fdata()).float()
    else:
        mask_fixed = None
        mask_moving = None

    H, W, D = img_fixed.shape

    # compute features and downsample (using average pooling)
    with torch.no_grad():

        features_fix, features_mov = extract_features(img_fixed=img_fixed,
                                                      img_moving=img_moving,
                                                      mind_r=mind_r,
                                                      mind_d=mind_d,
                                                      use_mask=use_mask,
                                                      mask_fixed=mask_fixed,
                                                      mask_moving=mask_moving)

        features_fix_smooth = F.avg_pool3d(features_fix,grid_sp,stride=grid_sp)
        features_mov_smooth = F.avg_pool3d(features_mov,grid_sp,stride=grid_sp)

        n_ch = features_fix_smooth.shape[1]

    movchunk = []
    fixchunk = []
    f0, f1 = features_fix_smooth.chunk(2, dim=2)
    f0_0, f0_1 = f0.chunk(2, dim=3)
    f0_0_0, f0_0_1 = f0_0.chunk(2, dim=4)
    fixchunk.append(f0_0_0)
    fixchunk.append(f0_0_1)
    f0_1_0, f0_1_1 = f0_1.chunk(2, dim=4)
    fixchunk.append(f0_1_0)
    fixchunk.append(f0_1_1)
    f1_0, f1_1 = f1.chunk(2, dim=3)
    f1_0_0, f1_0_1 = f1_0.chunk(2, dim=4)
    fixchunk.append(f1_0_0)
    fixchunk.append(f1_0_1)
    f1_1_0, f1_1_1 = f1_1.chunk(2, dim=4)
    fixchunk.append(f1_1_0)
    fixchunk.append(f1_1_1)
    del f0, f1, f0_0, f0_1, f0_0_0, f0_0_1, f0_1_0, f0_1_1, f1_0, f1_1, f1_0_0, f1_0_1, f1_1_0, f1_1_1

    x0, x1 = features_mov_smooth.chunk(2, dim=2)
    x0_0, x0_1 = x0.chunk(2, dim=3)
    x0_0_0, x0_0_1 = x0_0.chunk(2, dim=4)
    movchunk.append(x0_0_0)
    movchunk.append(x0_0_1)
    x0_1_0, x0_1_1 = x0_1.chunk(2, dim=4)
    movchunk.append(x0_1_0)
    movchunk.append(x0_1_1)
    x1_0, x1_1 = x1.chunk(2, dim=3)
    x1_0_0, x1_0_1 = x1_0.chunk(2, dim=4)
    movchunk.append(x1_0_0)
    movchunk.append(x1_0_1)
    x1_1_0, x1_1_1 = x1_1.chunk(2, dim=4)
    movchunk.append(x1_1_0)
    movchunk.append(x1_1_1)
    del x0, x1, x0_0, x0_1, x0_0_0, x0_0_1, x0_1_0, x0_1_1, x1_0, x1_1, x1_0_0, x1_0_1, x1_1_0, x1_1_1

    H, W, D = H//2, W//2, D//2

    flow_py = []
    for i in range(0, len(movchunk)):
        # compute correlation volume with SSD
        ssd, ssd_argmin = correlate(fixchunk[i], movchunk[i], disp_hw, grid_sp, (H, W, D), n_ch)
        # provide auxiliary mesh grid
        disp_mesh_t = F.affine_grid(disp_hw * torch.eye(3, 4).cuda().half().unsqueeze(0),
                                    (1, 1, disp_hw * 2 + 1, disp_hw * 2 + 1, disp_hw * 2 + 1),
                                    align_corners=True).permute(0, 4, 1, 2, 3).reshape(3, -1, 1)
        # perform coupled convex optimisation
        disp_soft = coupled_convex_mean(ssd, ssd_argmin, disp_mesh_t, grid_sp, (H, W, D))
        del ssd, ssd_argmin

        # if "ic" flag is set: make inverse consistent
        if ic:
            scale = torch.tensor([H // grid_sp - 1, W // grid_sp - 1, D // grid_sp - 1]).view(1, 3, 1, 1,
                                                                                              1).cuda().half() / 2

            ssd_, ssd_argmin_ = correlate(movchunk[i], fixchunk[i], disp_hw, grid_sp, (H, W, D), n_ch)

            disp_soft_ = coupled_convex_mean(ssd_, ssd_argmin_, disp_mesh_t, grid_sp, (H, W, D))
            disp_ice, _ = inverse_consistency((disp_soft / scale).flip(1), (disp_soft_ / scale).flip(1), iter=15)

            del ssd_, ssd_argmin_
            disp_hr = F.interpolate(disp_ice.flip(1) * scale * grid_sp, size=(H, W, D), mode='trilinear',
                                    align_corners=False)

        else:
            disp_hr = disp_soft

        flow_py.append(disp_hr)

    flow_0_0 = torch.cat((flow_py[0], flow_py[1]), dim=4)
    flow_0_1 = torch.cat((flow_py[2], flow_py[3]), dim=4)
    flow_1_0 = torch.cat((flow_py[4], flow_py[5]), dim=4)
    flow_1_1 = torch.cat((flow_py[6], flow_py[7]), dim=4)
    flow_0 = torch.cat((flow_0_0, flow_0_1), dim=3)
    flow_1 = torch.cat((flow_1_0, flow_1_1), dim=3)
    flow_p = torch.cat((flow_0, flow_1), dim=2)
    del flow_0_0, flow_0_1, flow_1_0, flow_1_1, flow_0, flow_1

    ####################################################################
    H, W, D = H * 2, W * 2, D * 2

    # run Adam instance optimisation
    if lambda_weight > 0:
        with torch.no_grad():
            patch_features_fix = F.avg_pool3d(features_fix,grid_sp_adam,stride=grid_sp_adam)
            patch_features_mov = F.avg_pool3d(features_mov,grid_sp_adam,stride=grid_sp_adam)
        del features_fix, features_mov

        #create optimisable displacement grid
        disp_lr = F.interpolate(flow_p,size=(H//grid_sp_adam,W//grid_sp_adam,D//grid_sp_adam),mode='trilinear',align_corners=False)

        net = nn.Sequential(nn.Conv3d(3,1,(H//grid_sp_adam,W//grid_sp_adam,D//grid_sp_adam),bias=False))
        net[0].weight.data[:] = disp_lr.float().cpu().data/grid_sp_adam
        del disp_lr
        net.cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=1)

        grid0 = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,H//grid_sp_adam,W//grid_sp_adam,D//grid_sp_adam),align_corners=False)

        #run Adam optimisation with diffusion regularisation and B-spline smoothing
        for iter in range(selected_niter):
            optimizer.zero_grad()

            disp_sample = F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(net[0].weight,3,stride=1,padding=1),3,stride=1,padding=1),3,stride=1,padding=1).permute(0,2,3,4,1)
            reg_loss = lambda_weight*((disp_sample[0,:,1:,:]-disp_sample[0,:,:-1,:])**2).mean()+\
            lambda_weight*((disp_sample[0,1:,:,:]-disp_sample[0,:-1,:,:])**2).mean()+\
            lambda_weight*((disp_sample[0,:,:,1:]-disp_sample[0,:,:,:-1])**2).mean()

            scale = torch.tensor([(H//grid_sp_adam-1)/2,(W//grid_sp_adam-1)/2,(D//grid_sp_adam-1)/2]).cuda().unsqueeze(0)
            grid_disp = grid0.view(-1,3).cuda().float()+((disp_sample.view(-1,3))/scale).flip(1).float()

            patch_mov_sampled = F.grid_sample(patch_features_mov.float(),grid_disp.view(1,H//grid_sp_adam,W//grid_sp_adam,D//grid_sp_adam,3).cuda(),align_corners=False,mode='bilinear')

            sampled_cost = (patch_mov_sampled-patch_features_fix).pow(2).mean(1)*12
            loss = sampled_cost.mean()
            (loss + reg_loss).backward()
            optimizer.step()

        fitted_grid = disp_sample.detach().permute(0,4,1,2,3)
        flow_p = F.interpolate(fitted_grid*grid_sp_adam,size=(H,W,D),mode='trilinear',align_corners=False)
        del fitted_grid, disp_sample

        if selected_smooth == 5:
            kernel_smooth = 5
            padding_smooth = kernel_smooth//2
            flow_p = F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(flow_p,kernel_smooth,padding=padding_smooth,stride=1),kernel_smooth,padding=padding_smooth,stride=1),kernel_smooth,padding=padding_smooth,stride=1)

        if selected_smooth == 3:
            kernel_smooth = 3
            padding_smooth = kernel_smooth//2
            flow_p = F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(flow_p,kernel_smooth,padding=padding_smooth,stride=1),kernel_smooth,padding=padding_smooth,stride=1),kernel_smooth,padding=padding_smooth,stride=1)

    x = flow_p[0, 0, :, :, :].cpu().half().data.numpy()
    y = flow_p[0, 1, :, :, :].cpu().half().data.numpy()
    z = flow_p[0, 2, :, :, :].cpu().half().data.numpy()
    displacements = np.stack((x, y, z), 3).astype(float)
    torch.cuda.empty_cache()

    return displacements, flow_p

def convex_adam(
    path_img_fixed: Union[Path, str],
    path_img_moving: Union[Path, str],
    mind_r: int = 1,
    mind_d: int = 2,
    lambda_weight: float = 1.25,
    grid_sp: int = 6,
    disp_hw: int = 4,
    selected_niter: int = 80,
    selected_niter_rigid: int = 500,
    selected_smooth: int = 0,
    grid_sp_adam: int = 2,
    ic: bool = True,
    use_mask: bool = False,
    path_fixed_mask: Optional[Union[Path, str]] = None,
    path_moving_mask: Optional[Union[Path, str]] = None,
    result_path: Union[Path, str] = './',
) -> None:
    """Coupled convex optimisation with adam instance optimisation"""
    img_fixed = torch.from_numpy(nib.load(path_img_fixed).get_fdata()).float()
    img_moving = torch.from_numpy(nib.load(path_img_moving).get_fdata()).float()

    displacements2, disp_flow2 = convex_adam_pt(
        img_fixed=img_fixed,
        img_moving=img_moving,
        mind_r=mind_r,
        mind_d=mind_d,
        lambda_weight=lambda_weight,
        grid_sp=grid_sp,
        disp_hw=disp_hw,
        selected_niter=selected_niter,
        selected_smooth=selected_smooth,
        grid_sp_adam=grid_sp_adam,
        ic=ic,
        use_mask=use_mask,
        path_fixed_mask=path_fixed_mask,
        path_moving_mask=path_moving_mask,
    )

    flow = disp_flow2
    disp_out = displacements2

    return flow, disp_out

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-dp", '--datasets_path', type = str, default = "/input/")
    parser.add_argument("-dp", '--datasets_path', type=str, default="./valt2")
    parser.add_argument('--mind_r', type=int, default=1)
    parser.add_argument('--mind_d', type=int, default=2)
    parser.add_argument('--lambda_weight', type=float, default=1.25) #1.25
    parser.add_argument('--grid_sp', type=int, default=4)
    parser.add_argument('--disp_hw', type=int, default=4)
    parser.add_argument('--selected_niter', type=int, default=15)
    parser.add_argument('--selected_smooth', type=int, default=5)
    parser.add_argument('--grid_sp_adam', type=int, default=2)
    parser.add_argument('--ic', choices=('True','False'), default='True')
    parser.add_argument('--use_mask', choices=('True','False'), default='False')
    parser.add_argument('--path_mask_fixed', type=str, default=None)
    parser.add_argument('--path_mask_moving', type=str, default=None)
    # parser.add_argument('--result_path', type=str, default='/output/')
    parser.add_argument('--result_path', type=str, default='./result')

    args = parser.parse_args()

    fixed_mod = '0000'
    moving_mod = '0002'
    fusion_mod = '0001'

    # select inference cases
    list_case = sorted([k.split('_')[1] for k in os.listdir(args.datasets_path) if f'{moving_mod}.nii.gz' in k])
    print(f"Number total cases: {len(list_case)}")

    for case in list_case:
        torch.cuda.synchronize()
        t0 = time.time()
        # Load image using SimpleITK
        fix_path = os.path.join(args.datasets_path, f"ReMIND2Reg_{case}_{fixed_mod}.nii.gz")
        mov_path = os.path.join(args.datasets_path, f"ReMIND2Reg_{case}_{moving_mod}.nii.gz")
        fusion_path = os.path.join(args.datasets_path, f"ReMIND2Reg_{case}_{fusion_mod}.nii.gz")
        img_fixed = torch.from_numpy(nib.load(fix_path).get_fdata()).float()
        img_moving = torch.from_numpy(nib.load(mov_path).get_fdata()).float()

        flow, disp_out = convex_adam(
                            path_img_fixed=fix_path,
                            path_img_moving=mov_path,
                            mind_r=args.mind_r,
                            mind_d=args.mind_d,
                            lambda_weight=args.lambda_weight,
                            grid_sp=args.grid_sp,
                            disp_hw=args.disp_hw,
                            selected_niter=args.selected_niter,
                            selected_smooth=args.selected_smooth,
                            grid_sp_adam=args.grid_sp_adam,
                            ic=(args.ic == 'True'),
                            use_mask=(args.use_mask == 'True'),
                            path_fixed_mask=args.path_mask_fixed,
                            path_moving_mask=args.path_mask_moving,
                            result_path=args.result_path
                        )

        flow2, disp_out2 = convex_adam(
            path_img_fixed=fix_path,
            path_img_moving=mov_path,
            mind_r=args.mind_r,
            mind_d=args.mind_d,
            lambda_weight=args.lambda_weight,
            grid_sp=args.grid_sp + 2,
            disp_hw=args.disp_hw,
            selected_niter=args.selected_niter,
            selected_smooth=args.selected_smooth,
            grid_sp_adam=args.grid_sp_adam,
            ic=(args.ic == 'True'),
            use_mask=(args.use_mask == 'True'),
            path_fixed_mask=args.path_mask_fixed,
            path_moving_mask=args.path_mask_moving,
            result_path=args.result_path
        )

        flow3, disp_out3 = convex_adam(
            path_img_fixed=fix_path,
            path_img_moving=mov_path,
            mind_r=args.mind_r,
            mind_d=args.mind_d,
            lambda_weight=args.lambda_weight,
            grid_sp=args.grid_sp - 2,
            disp_hw=args.disp_hw,
            selected_niter=args.selected_niter,
            selected_smooth=args.selected_smooth,
            grid_sp_adam=args.grid_sp_adam,
            ic=(args.ic == 'True'),
            use_mask=(args.use_mask == 'True'),
            path_fixed_mask=args.path_mask_fixed,
            path_moving_mask=args.path_mask_moving,
            result_path=args.result_path
        )

        disp_p, flow_p = convex_adam_pyramid(
            img_fixed=img_fixed,
            img_moving=img_moving,
            mind_r=args.mind_r,
            mind_d=args.mind_d,
            lambda_weight=args.lambda_weight,
            grid_sp=1,
            disp_hw=args.disp_hw,
            selected_niter=args.selected_niter,
            selected_smooth=args.selected_smooth,
            grid_sp_adam=args.grid_sp_adam,
            ic=True,
            use_mask=False,
            path_fixed_mask=None,
            path_moving_mask=None,
        )

        if os.path.exists(fusion_path):
            img_fu = torch.from_numpy(nib.load(fusion_path).get_fdata()).float()
            flowfu, disp_outfu = convex_adam(
                path_img_fixed=fix_path,
                path_img_moving=fusion_path,
                mind_r=args.mind_r,
                mind_d=args.mind_d,
                lambda_weight=args.lambda_weight,
                grid_sp=args.grid_sp,
                disp_hw=args.disp_hw,
                selected_niter=args.selected_niter,
                selected_smooth=args.selected_smooth,
                grid_sp_adam=args.grid_sp_adam,
                ic=(args.ic == 'True'),
                use_mask=(args.use_mask == 'True'),
                path_fixed_mask=args.path_mask_fixed,
                path_moving_mask=args.path_mask_moving,
                result_path=args.result_path
            )

            flow2fu, disp_out2fu = convex_adam(
                path_img_fixed=fix_path,
                path_img_moving=fusion_path,
                mind_r=args.mind_r,
                mind_d=args.mind_d,
                lambda_weight=args.lambda_weight,
                grid_sp=args.grid_sp + 2,
                disp_hw=args.disp_hw,
                selected_niter=args.selected_niter,
                selected_smooth=args.selected_smooth,
                grid_sp_adam=args.grid_sp_adam,
                ic=(args.ic == 'True'),
                use_mask=(args.use_mask == 'True'),
                path_fixed_mask=args.path_mask_fixed,
                path_moving_mask=args.path_mask_moving,
                result_path=args.result_path
            )

            flow3fu, disp_out3fu = convex_adam(
                path_img_fixed=fix_path,
                path_img_moving=fusion_path,
                mind_r=args.mind_r,
                mind_d=args.mind_d,
                lambda_weight=args.lambda_weight,
                grid_sp=args.grid_sp - 2,
                disp_hw=args.disp_hw,
                selected_niter=args.selected_niter,
                selected_smooth=args.selected_smooth,
                grid_sp_adam=args.grid_sp_adam,
                ic=(args.ic == 'True'),
                use_mask=(args.use_mask == 'True'),
                path_fixed_mask=args.path_mask_fixed,
                path_moving_mask=args.path_mask_moving,
                result_path=args.result_path
            )

            disp_pfu, flow_pfu = convex_adam_pyramid(
                img_fixed=img_fixed,
                img_moving=img_fu,
                mind_r=args.mind_r,
                mind_d=args.mind_d,
                lambda_weight=args.lambda_weight,
                grid_sp=1,
                disp_hw=args.disp_hw,
                selected_niter=args.selected_niter,
                selected_smooth=args.selected_smooth,
                grid_sp_adam=args.grid_sp_adam,
                ic=True,
                use_mask=False,
                path_fixed_mask=None,
                path_moving_mask=None,
            )

        affine = nib.load(fix_path).affine
        reg_model = registerSTModel([256, 256, 256], 'nearest').cuda()

        if os.path.exists(fusion_path):
            flow_gs4 = torch.max(flow, flowfu)  ##max
            flow_gs6 = torch.max(flow2, flow2fu)  ##max
            flow_gs2 = torch.max(flow3, flow3fu)  ##max
            flow_gs1 = torch.max(flow_p, flow_pfu)  ##max
            flow = 0.27 * flow_gs2 + 0.36 * flow_gs6 + 0.27 * flow_gs4 + 0.1 * flow_gs1
            out_t = reg_model(img_moving.cuda().view(1, 1, 256, 256, 256), flow)
            out_nii_t = nib.Nifti1Image(out_t.view(256, 256, 256).cpu().detach().numpy(), affine)

            disp_out_gs4 = np.maximum(disp_out, disp_outfu)  ##max
            disp_out_gs6 = np.maximum(disp_out2, disp_out2fu)  ##max
            disp_out_gs2 = np.maximum(disp_out3, disp_out3fu)  ##max
            disp_out_gs1 = np.maximum(disp_p, disp_pfu)  ##max
            disp_out = 0.27 * disp_out_gs4 + 0.36 * disp_out_gs6 + 0.27 * disp_out_gs2 + 0.1 * disp_out_gs1
            disp_nii_t = nib.Nifti1Image(disp_out, affine)

            mov = mov_path.split("_", 1)[1].split(".", 1)[0]
            fix = fix_path.split("_", 1)[1].split(".", 1)[0]

            nib.save(out_nii_t, os.path.join(args.result_path, 'out_' + fix + '_' + mov + '.nii.gz'))
            nib.save(disp_nii_t, os.path.join(args.result_path, 'disp_' + fix + '_' + mov + '.nii.gz'))
        else:
            flow_gs4 = flow
            flow_gs6 = flow2
            flow_gs2 = flow3
            flow_gs1 = flow_p
            flow = 0.27 * flow_gs2 + 0.36 * flow_gs6 + 0.27 * flow_gs4 + 0.1 * flow_gs1
            out_t = reg_model(img_moving.cuda().view(1, 1, 256, 256, 256), flow)
            out_nii_t = nib.Nifti1Image(out_t.view(256, 256, 256).cpu().detach().numpy(), affine)

            disp_out_gs4 = disp_out
            disp_out_gs2 = disp_out2
            disp_out_gs6 = disp_out3
            disp_out_gs1 = disp_p
            disp_out = 0.27 * disp_out_gs4 + 0.36 * disp_out_gs6 + 0.27 * disp_out_gs2 + 0.1 * disp_out_gs1
            disp_nii_t = nib.Nifti1Image(disp_out, affine)

            mov = mov_path.split("_", 1)[1].split(".", 1)[0]
            fix = fix_path.split("_", 1)[1].split(".", 1)[0]

            nib.save(out_nii_t, os.path.join(args.result_path,
                                               'out_' + fix + '_' + mov + '.nii.gz'))
            nib.save(disp_nii_t, os.path.join(args.result_path,
                                              'disp_' + fix + '_' + mov + '.nii.gz'))

        torch.cuda.synchronize()
        t1 = time.time()
        case_time = t1 - t0
        print('case time: ', case_time)

        torch.cuda.empty_cache()
