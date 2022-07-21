import torch
from tqdm import tqdm
from .rotations import *
import time

def power_iteration(M,max_iters=10,eps=1e-3):
    """
        M: (*,N,N)
    """
    N = M.shape[-1]
    assert M.shape[-1] == M.shape[-2]
    q = torch.randn(M.shape[:-2]+(4,),dtype=M.dtype,device=M.device) # (*,4)
    q = q/torch.norm(q,p=2,dim=-1,keepdim=True)
    q = q.unsqueeze(-1)
    for _ in range(max_iters):
        q2 = M@q
        q2 = q2 / torch.norm(q2,p=2,keepdim=True)
        if (q-q2).pow(2).squeeze(-1).sum(-1).max()<=eps:
            break
        q = q2
    return q2.squeeze(-1)

def get_neibor_graph(rel_rot_inds,n):
    matrix = torch.zeros((n,n),dtype=torch.uint8)
    for inds in rel_rot_inds:
        idx1,idx2 = int(inds[0]),int(inds[1])
        matrix[idx1,idx2] = 1
        matrix[idx2,idx1] = 1
    return matrix

def get_spanning_tree(graph):
    root_id = int(torch.argmax(torch.sum(graph,dim=1)))
    n = graph.shape[0]
    tree = {}
    queue = [root_id]
    matrix_copy = torch.clone(graph)
    matrix_copy[:,root_id] = 0
    while len(queue) != 0:
        cur = queue[0]
        tree[cur] = [i for i in range(n) if matrix_copy[cur,i]==1]
        del queue[0]
        queue += tree[cur]
        matrix_copy[:,tree[cur]] = 0
    return root_id,tree

def get_rotations_by_inds(rel_rots,rel_rot_inds):
    rots = {}
    for i,inds in enumerate(rel_rot_inds):
        id1,id2 = inds
        rots[f'{id1}_{id2}'] = rel_rots[i]
        rots[f'{id2}_{id1}'] = quaternion_invert(rel_rots[i])
    return rots


def initialize(root_id,tree,rel_rots,n,device):
    # rot_pred = {}
    rot_pred = torch.zeros((n,4),dtype=torch.float32,device=device)
    rot_pred[root_id,0] = 1
    
    # first in first out
    queue = [root_id]
    while len(queue)!=0:
        ind_cur = queue[0]
        R_cur = rot_pred[ind_cur]
        for ind in tree[ind_cur]:
            rot_pred[ind] = quaternion_multiply(rel_rots[f'{ind_cur}_{ind}'],R_cur)
        del queue[0]
        queue += tree[ind_cur]
    return rot_pred

class SingleRotationSolver:
    def __init__(self,method='L1',steps=5,pw_iter=10):
        self.steps = steps
        self.pw_iter = pw_iter
        self.method = method

    
    def solve(self,rotations):
        """ 
            rotations: (N,4)
        """
        A = (rotations.unsqueeze(-1)@rotations.unsqueeze(-2)).mean(0) # 4,4
        s = power_iteration(A,self.pw_iter) # 4
        if self.method=='L1':
            for _ in range(self.steps):
                v = quaternion_to_axis_angle(quaternion_multiply(rotations,quaternion_invert(s))) # N,3
                norms = torch.norm(v,p=2,dim=-1,keepdim=True)
                v_normalized = v/norms
                if (norms==0.).sum()>0:
                    break
                delta = axis_angle_to_quaternion(v_normalized.sum(dim=0)/(1/norms).sum()) # 4
                s = quaternion_multiply(delta,s)
            
        return s
        
                
class MultipleRotationSolver:
    def __init__(self,method='L1',steps=3,sweeps=3,pw_iter=5,progress_bar=True):
        self.single_solver = SingleRotationSolver(method,steps=steps,pw_iter=pw_iter)
        self.sweeps = sweeps
        self.progress_bar = progress_bar
    
    def solve(self,rel_rots,rel_rot_inds):
        device = rel_rots.device
        rel_rots = matrix_to_quaternion(rel_rots)
        rel_rots = get_rotations_by_inds(rel_rots,rel_rot_inds)
        n = rel_rot_inds.max()+1

        neighbor_graph = get_neibor_graph(rel_rot_inds,n)
        root_id, tree = get_spanning_tree(neighbor_graph)

        rots = initialize(root_id,tree,rel_rots,n,device)
        if self.progress_bar:
            bar = tqdm(range(self.sweeps),ncols=80)
        else:
            bar = range(self.sweeps)

        for _ in bar:
            t1 = 0
            t2 = 0
            t3 = 0
            for ind in range(n):
                if ind == root_id:
                    continue
                t= time.time()
                neighbor_inds = torch.where(neighbor_graph[ind]!=0)[0]
                neighbor_inds = neighbor_inds.tolist()
                t1 += time.time()-t
                t=time.time()
                q1 = torch.stack([rel_rots[f'{nb_ind}_{ind}'] for nb_ind in neighbor_inds])
                q2 = torch.stack([rots[nb_ind] for nb_ind in neighbor_inds])
                rot_preds = quaternion_multiply(q1,q2)
                # rot_preds = torch.stack([(quaternion_multiply(rel_rots[f'{nb_ind}_{ind}'],rots[nb_ind])) for nb_ind in neighbor_inds])
                t2 += time.time()-t
                if len(rot_preds)==1:
                    continue
                t = time.time()
                rot_averaged = self.single_solver.solve(rot_preds)
                t3 += time.time()-t
                rots[ind] = rot_averaged
            # print(t1,t2,t3)
        return rots