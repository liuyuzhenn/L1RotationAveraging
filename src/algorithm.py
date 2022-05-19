import cv2
from tqdm import tqdm
import numpy as np
from .utils import *
import time


def get_neibor_graph(rel_rot_inds,n):
    matrix = np.zeros((n,n),dtype=int)
    for inds in rel_rot_inds:
        matrix[inds[0],inds[1]] = 1
        matrix[inds[1],inds[0]] = 1
    return matrix

def get_spanning_tree(graph):
    root_id = np.argmin(np.sum(graph,axis=1))
    n = graph.shape[0]
    tree = {}
    queue = [root_id]
    matrix_copy = np.copy(graph)
    matrix_copy[:,root_id] = 0
    while len(queue) != 0:
        cur = queue[0]
        tree[cur] = [i for i in range(n) if matrix_copy[cur,i]==1]
        del queue[0]
        queue += tree[cur]
        matrix_copy[:,tree[cur]] = 0
    return root_id,tree


def proj2SO3(R):
    u,s,vt = np.linalg.svd(R)
    uvt = u@vt
    if np.linalg.det(uvt)>=0:
        return uvt
    else:
        s_ = np.diag([1.0,1.0,-1.0])
        return u@s_@vt


def chordal_L2_mean(rots):
    '''Compute L2 mean under the chordal metric, i.e. Frobenius norm'''
    rot_mean = np.mean(rots,axis=0)
    return proj2SO3(rot_mean)


def initialize(root_id,tree,rel_rots):
    rot_pred = {}
    rot_pred[root_id] = np.identity(3,dtype=float)
    
    # first in first out
    queue = [root_id]
    while len(queue)!=0:
        ind_cur = queue[0]
        R_cur = rot_pred[ind_cur]
        for ind in tree[ind_cur]:
            rot_pred[ind] = rel_rots[f'{ind_cur}_{ind}']@R_cur
        del queue[0]
        queue += tree[ind_cur]
    return rot_pred

def align_rotation(rots_pred,rots_gt):
    n = len(rots_pred)
    rots_diff = [rots_pred[i].transpose()@rots_gt[i] for i in range(n)]
    rots_diff_SO3 = chordal_L2_mean(rots_diff)
    rots_aligned = [rots_pred[i]@rots_diff_SO3 for i in range(n)]
    return rots_aligned

def multiple_rotation_averaging(rel_rots,rel_rot_inds,n,steps=1,max_iterations=1000,eps=1e-3):
    '''Rotation averaging algorithm proposed in 
        "L1 rotation averaging using the Weiszfeld algorithm"'''
    neighbor_graph = get_neibor_graph(rel_rot_inds,n)
    root_id, tree = get_spanning_tree(neighbor_graph)

    # Initialization
    rots = initialize(root_id,tree,rel_rots)

    # Sweep
    bar = tqdm(range(max_iterations),ncols=80)
    for _ in bar:
        stop = True
        for ind in range(n):
            if ind == root_id:
                continue
            neighbor_inds = np.where(neighbor_graph[ind]!=0)[0]
            rot_preds = [rel_rots[f'{nb_ind}_{ind}']@rots[nb_ind] for nb_ind in neighbor_inds]
            if len(rot_preds)==1:
                continue
            rot_averaged = single_rotation_averaging(rot_preds,steps=steps)
            rots[ind] = rot_averaged

            dtheta = np.arccos(np.trace((rot_averaged.transpose()@rots[ind]-1)/2))/np.pi*180
            if dtheta>=eps:
                stop=False
        # Converged
        if stop:
            break
    return rots

def single_rotation_averaging(rots,steps=100):
    # 1. Initialize with chordal L2 mean
    S = chordal_L2_mean(rots)
    stop=False
    for _ in range(steps):
        # 2. Compute vi
        v = np.zeros((3,1),dtype=float)
        v_norm_inv = 0.
        for r in rots:
            vi = cv2.Rodrigues(r@S.transpose())[0]
            norm = np.linalg.norm(vi)
            if norm==0:
                stop=True
                break
            v += vi/norm
            v_norm_inv += 1/norm
        if stop:
            break
        # 3. Weiszfeld step
        delta = v/v_norm_inv
        # 4. Update R
        exp_delta = cv2.Rodrigues(delta)[0]
        S = exp_delta@S
    return S


if __name__ == '__main__':
    rel_rots,rel_rot_inds,rots_gt,valid_inds = load_Alamo('./data/Alamo.mat')

    t1 = time.time()
    rots_pred = multiple_rotation_averaging(rel_rots,rel_rot_inds,rots_gt.shape[0],
                                            steps=10,max_iterations=2,eps=1e-1)
    t2 = time.time()

    rots_pred = [rots_pred[ind] for ind in valid_inds]
    rots_gt = [rots_gt[ind] for ind in valid_inds]

    rots_aligned = align_rotation(rots_pred,rots_gt)
    errors = get_errors(rots_gt,rots_aligned)
    n = len(errors)
    if n%2==0:
        error = (errors[n//2]+errors[n//2-1])/2/np.pi*180
    else:
        error = errors[n//2]/np.pi*180
    print('Median error is {:.3f} degree'.format(error))
    print('Time cost is {:.3f} s'.format(t2-t1))

    # N = get_neibor_graph(rel_rot_inds,rots_gt.shape[0])
    # n = N.shape[0]
    # t1 = time.time()
    # root_id,tree = get_spanning_tree(N)
    # t2 = time.time()
    # print('Building a tree with {} nodes takes {:.3f} ms'.format(N.shape[0],(t2-t1)*1000))
    # rots_pred = initialize(root_id,tree,rel_rots)
    
    # rots_pred = [rots_pred[ind] for ind in valid_inds]
    # rots_gt = [rots_gt[ind] for ind in valid_inds]
    # rots_aligned = align_rotation(rots_pred,rots_gt)
    # error = get_median_error(rots_gt,rots_aligned)
    # print(error)