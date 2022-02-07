from matplotlib.pyplot import inferno
import numpy as np
import torch
import random

from torch._C import dtype
torchpi = torch.acos(torch.zeros(1)).item() * 2

# original implementation from Chatterjee's ICCV13 paper: CompareRotationGraph.m

def R2w(R):
    w = torch.stack((R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0]- R[0,1])) / 2
    s = torch.norm(w)
    if s:
        w = w / s * torch.atan2(s,(torch.trace(R) - 1) / 2)
    return w

def w2R(w):
    omega = torch.norm(w)
    if omega:
        n = w / omega
        s = torch.sin(omega)
        c = torch.cos(omega)
        cc = 1- c
        n1 = n[0]
        n2 = n[1]
        n3 = n[2]
        n12cc=n1*n2*cc
        n23cc=n2*n3*cc
        n31cc=n3*n1*cc
        n1s=n1*s              
        n2s=n2*s              
        n3s=n3*s
        R = torch.zeros(3,3)
        R[0,0]=c+n1*n1*cc
        R[0,1]=n12cc-n3s      
        R[0,2]=n31cc+n2s
        R[1,0]=n12cc+n3s     
        R[1,1]=c+n2*n2*cc     
        R[1,2]=n23cc-n1s
        R[2,0]=n31cc-n2s      
        R[2,1]=n23cc+n1s      
        R[2,2]=c+n3*n3*cc
    else:
        R = torch.eye(3)
    R = R.to(torch.float64)
    return R

def compare_rot_graph(R1, R2, method="median"):
    sigma2 = (5 * torchpi / 180 ) *(5 * torchpi / 180 ) 
    N = R1.shape[0]
    Emeanbest = float("Inf")
    E = torch.zeros(3)
    Ebest = E.clone()
    e = torch.zeros(N,1)
    ebest = e
    l = [39,38,1,93]
    for i in range(4):
        j = random.randint(0,N-1)
        # print(j)
        #j = i *2 
        # j = l[i]
        R = R1[j,:,:].clone().t()
        for k in range(N):
            R1[k,:,:] = torch.mm (R1[k,:,:], R)
        R = R2[j,:,:].clone().t()
        for k in range(N):
            R2[k,:,:] = torch.mm (R2[k,:,:], R)
        W = torch.zeros(N,3)
        d = float("Inf")
        count = 1
        while(d>1e-5 and count < 20):
            for k in range(N):
                W[k,:] = R2w(torch.mm(R2[k,:,:].t(),R1[k,:,:]))
            
            if method == "mean":
                w = torch.mean(W,0).values
                d = torch.norm(w)
                R = w2R(w)
            elif method == "median":
                w = torch.median(W,0).values
                d = torch.norm(w)
                R = w2R(w)
            elif method == "robustmean":
                w = 1 / torch.sqrt( torch.sum(W * W, 1) + sigma2)
                w = w/torch.sum(w)
                w = torch.mean( w.repeat(1,3) * W)
                d = torch.norm(w)
                R = w2R(w)
            for k in range(N):
                R2[k,:,:] = torch.mm (R2[k,:,:], R.to(torch.float))
            count = count + 1
        for k in range(N):
            # e[k,0] = torch.acos(torch.max(torch.min((torch.sum(R1[k,0,:]*R2[k,0,:].t()) 
            # + torch.sum(R1[k,1,:]*R2[k,1,:].t())+ torch.sum(R1[k,2,:]*R2[k,2,:].t())-1)/2,1),-1))
            now = (torch.sum(R1[k,0,:]*R2[k,0,:].t()) + torch.sum(R1[k,1,:]*R2[k,1,:].t())+ torch.sum(R1[k,2,:]*R2[k,2,:].t())-1)/2
            if now > 1:
                now = torch.tensor(1,dtype=torch.float32)
            if now < -1:
                now = torch.tensor(-1,dtype=torch.float32)
            e[k,0] = torch.acos(now)
        e = e * 180 / torchpi
        E = torch.stack([torch.mean(e),torch.median(e), torch.sqrt(torch.mm(e.t(),e)/len(e))[0,0]])
        if E[1] < Emeanbest:
            Ebest = E
            Emeanbest = E[2]
    
    E_mean, E_median, E_var = Ebest[0].item(), Ebest[1].item(), Ebest[2].item()
    return E_mean, E_median, E_var

def read_rot(filename):
    with open(filename, "r") as fin:
        lines = fin.readlines()
    lines = [x.split()[1:] for x in lines]
    lines = [ [float(y) for y in x] for x in lines]
    R = np.asarray(lines)
    R = R.reshape((-1,3,3))
    return R
if __name__ == "__main__":
    # file1 = "./data/R1.txt"
    # file2 = "./data/R2.txt"
    # R1 = read_rot(file1)
    # R2 = read_rot(file2)
    # R1 = torch.from_numpy(R1).to(torch.float64)
    # R2 = torch.from_numpy(R2).to(torch.float64)
    # e = compare_rot_graph(R1,R2)
    # print(e)
    # 
    w = torch.rand(3)
    R = w2R(w)
    print(R)
