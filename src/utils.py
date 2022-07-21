import numpy as np
import torch
from .rotations import *
from .averaging import *
from scipy.io import loadmat

def load_Alamo(path):
    d = loadmat(path)
    rel_rots = d['RRAlamo'].transpose(2,0,1).astype(float)
    valid_inds = d['validIndex'].reshape(-1)-1
    gt_rot = d['RgtAlamo'].transpose(2,0,1).astype(float)
    rel_rot_inds = d['IIAlamo'].transpose()-1
    return rel_rots,rel_rot_inds.astype(int),gt_rot,valid_inds.astype(int)

def load_Ellis(path):
    d = loadmat(path)
    rel_rots = d['RR'].transpose(2,0,1).astype(float)
    rel_rot_inds = d['I'].transpose()-1
    gt_rot = d['Rgt'].transpose(2,0,1).astype(float)
    valid_inds = np.arange(0,gt_rot.shape[0])
    return rel_rots,rel_rot_inds,gt_rot,valid_inds.astype(int)

def get_errors(gt_rots,pred_rots,valid_inds):
    errors = []
    n = len(gt_rots)
    for ind in range(n):
        if ind not in valid_inds:
            continue
        gt_R = gt_rots[ind]
        pred_R = pred_rots[ind]
        tr = np.trace(gt_R@pred_R.transpose())
        errors.append(np.arccos((tr-1)/2)*180/np.pi)
    return errors



def get_rotations_by_inds(rel_rots,rel_rot_inds):
    rots = {}
    for i,inds in enumerate(rel_rot_inds):
        id1,id2 = inds
        rots[f'{id1}_{id2}'] = rel_rots[i]
        rots[f'{id2}_{id1}'] = rel_rots[i].transpose()
    return rots


def evaluate(rot_pred,rot_gt,valid_inds):
    n = len(rot_gt)
    mask = torch.zeros((n,),dtype=bool)
    mask[valid_inds.long()] = 1
    rot_pred_ = rot_pred[mask]
    rot_gt_ = rot_gt[mask]

    # align
    diff = quaternion_multiply(quaternion_invert(rot_pred_),rot_gt_) #
    solver = SingleRotationSolver('L1',steps=5,pw_iter=5)

    diff_avg = solver.solve(diff).unsqueeze(0).repeat(rot_gt_.shape[0],1)

    rot_pred_ = quaternion_multiply(rot_pred_,diff_avg) # 

    diff = quaternion_multiply(rot_pred_,quaternion_invert(rot_gt_))
    angles = torch.acos(diff[:,0])*2/np.pi*180
    return angles
