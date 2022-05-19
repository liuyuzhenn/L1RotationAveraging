import numpy as np
from scipy.io import loadmat

def load_Alamo(path):
    d = loadmat(path)
    rel_rots = d['RRAlamo'].transpose(2,0,1).astype(float)
    valid_inds = d['validIndex'].reshape(-1)-1
    gt_rot = d['RgtAlamo'].transpose(2,0,1).astype(float)
    rel_rot_inds = d['IIAlamo'].transpose()-1
    rel_rots = get_rotations_by_inds(rel_rots,rel_rot_inds)
    return rel_rots,rel_rot_inds,gt_rot,valid_inds

def load_Ellis(path):
    d = loadmat(path)
    rel_rots = d['RR'].transpose(2,0,1).astype(float)
    rel_rot_inds = d['I'].transpose()-1
    gt_rot = d['Rgt'].transpose(2,0,1).astype(float)
    valid_inds = np.arange(0,gt_rot.shape[0])
    rel_rots = get_rotations_by_inds(rel_rots,rel_rot_inds)
    return rel_rots,rel_rot_inds,gt_rot,valid_inds

def get_errors(gt_rots,pred_rots):
    errors = []
    n = len(gt_rots)
    for ind in range(n):
        gt_R = gt_rots[ind]
        pred_R = pred_rots[ind]
        tr = np.trace(gt_R@pred_R.transpose())
        errors.append(np.arccos((tr-1)/2)*180/np.pi)
    # errors.sort()
    return errors
    # n = len(errors)
    # if n%2==0:
    #     return (errors[n//2]+errors[n//2-1])/2/np.pi*180
    # else:
    #     return errors[n//2]/np.pi*180


def get_rotations_by_inds(rel_rots,rel_rot_inds):
    rots = {}
    for i,inds in enumerate(rel_rot_inds):
        id1,id2 = inds
        rots[f'{id1}_{id2}'] = rel_rots[i]
        rots[f'{id2}_{id1}'] = rel_rots[i].transpose()
    return rots

if __name__ == '__main__':
    rel_rots,rel_rot_inds,gt_rot,valid_inds = load_Alamo('./data/Alamo.mat')

    id1,id2 = rel_rot_inds[0]
    R12_rel = rel_rots[0]
    R1 = gt_rot[id1]
    R2 = gt_rot[id2]
    R12 = R2@R1.transpose()
    print(R12_rel)
    print(R12)