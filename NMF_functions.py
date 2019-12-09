import torch
import torch.nn as nn
SEloss = nn.MSELoss(reduction = 'sum')

class NMF_algorithim(nn.Module):
    ''' implements ARD NMF from https://arxiv.org/pdf/1111.6085.pdf '''
    def __init__(self,Beta,H_prior,W_prior):
        super(NMF_algorithim, self).__init__()
        # Beta paramaterizes the objective function
        # Beta = 1 induces a poisson objective
        # Beta = 2 induces a gaussian objective
        # Priors on the component matrices are Exponential (L1) and half-normal (L2)

        if Beta == 1 and H_prior == 'L1' and W_prior == 'L1' :
            self.update_W = update_W_poisson_L1
            self.update_H = update_H_poisson_L1
            self.lambda_update = update_lambda_L1

        elif Beta == 1 and H_prior == 'L1' and W_prior == 'L2':
            self.update_W = update_W_poisson_L2
            self.update_H = update_H_poisson_L1
            self.lambda_update = update_lambda_L2_L1

        elif Beta == 1 and H_prior == 'L2' and W_prior == 'L1':
            self.update_W = update_W_poisson_L1
            self.update_H = update_H_poisson_L2
            self.lambda_update = update_lambda_L1_L2

        elif Beta == 1 and H_prior == 'L2' and W_prior == 'L2':
            self.update_W = update_W_poisson_L2
            self.update_H = update_H_poisson_L2
            self.lambda_update = update_lambda_L2

        if Beta == 2 and H_prior == 'L1' and W_prior == 'L1':
            self.update_W = update_W_gaussian_L1
            self.update_H = update_H_gaussian_L1
            self.lambda_update = update_lambda_L1

        elif Beta == 2 and H_prior == 'L1' and W_prior == 'L2':
            self.update_W = update_W_gaussian_L2
            self.update_H = update_H_gaussian_L1
            self.lambda_update = update_lambda_L2_L1

        elif Beta == 2 and H_prior == 'L2' and W_prior == 'L1':
            self.update_W = update_W_gaussian_L1
            self.update_H = update_H_gaussian_L2
            self.lambda_update = update_lambda_L1_L2

        elif Beta == 2 and H_prior == 'L2' and W_prior == 'L2':
            self.update_W = update_W_gaussian_L2
            self.update_H = update_H_gaussian_L2
            self.lambda_update = update_lambda_L2

    def forward(self,W, H, V, lambda_, C, b0, eps_, phi):
        h_ = self.update_H(H, W, lambda_, phi, V, eps_)
        w_ = self.update_W(h_, W, lambda_, phi, V, eps_)
        lam_ = self.lambda_update(w_,h_,b0,C,eps_)
        return h_, w_,lam_



def beta_div(Beta,V,W,H,eps_):
    V_ap = torch.matmul(W, H).type(V.dtype) + eps_.type(V.dtype)
    if Beta == 2:
        return SEloss(V,V_ap)/2
    if Beta == 1:
        lr = torch.log(torch.div(V, V_ap))
        return torch.sum( ( (V*lr) + V_ap) - V)

def calculate_objective_function(Beta,V,W,H,lambda_,C, eps_,phi,K):
    loss = beta_div(Beta,V,W,H,eps_)
    cst = (K*C)*(1.0-torch.log(C))
    return torch.pow(phi,-1)*loss + (C*torch.sum(torch.log(lambda_ * C))) + cst

def update_H_poisson_L1(H, W, lambda_, phi, V, eps_):
    #beta = 1 gamma(beta) = 1
    denom = torch.sum(W, 0) + torch.div(phi, lambda_) + eps_
    V_ap = torch.matmul(W, H) + eps_
    V_res = torch.div(V, V_ap)
    update = torch.div(torch.matmul(W.transpose(1,0), V_res), denom.reshape(-1,1))
    return H * update

def update_H_poisson_L2(H,W,lambda_,phi,V, eps_):
    #beta = 1 zeta(beta) = 1/2
    denom = torch.sum(W,0).reshape(-1,1) + torch.div(phi*H, lambda_.reshape(-1,1)) + eps_
    V_ap = torch.matmul(W, H) + eps_
    update = torch.pow(torch.div(torch.matmul(W.transpose(0,1), torch.div(V, V_ap)), denom),0.5)
    return H * update

def update_H_gaussian_L1(H,W,lambda_,phi,V,eps_):
    #beta = 2 gamma(beta) = 1
    V_ap = torch.matmul(W, H) + eps_
    denom = torch.matmul(W.transpose(0,1),V_ap) + torch.div(phi, lambda_ ).reshape(-1,1) + eps_
    update = torch.div(torch.matmul(W.transpose(0,1),V),denom)
    return H * update

def update_H_gaussian_L2(H,W,lambda_,phi,V,eps_):
    #beta = 2 zeta(beta) = 1
    denom = torch.matmul(W.transpose(0,1).type(V.dtype),torch.matmul(W, H).type(V.dtype) + eps_) + torch.div(phi * H, lambda_.reshape(-1,1)).type(V.dtype) + eps_
    update = torch.div(torch.matmul(W.transpose(0,1).type(V.dtype),V),denom)
    return H * update.type(torch.float32)

def update_W_poisson_L1(H, W, lambda_, phi, V, eps_):
    #beta = 1 gamma(beta) = 1
    denom = torch.sum(H, 1) + torch.div(phi, lambda_ ) + eps_
    V_ap = torch.matmul(W, H) + eps_
    V_res = torch.div(V, V_ap)
    update = torch.div(torch.matmul(V_res, H.transpose(0,1)), denom)
    return W * update

def update_W_poisson_L2(H,W,lambda_,phi,V,eps_):
    # beta = 1 zeta(beta) = 1/2
    V_ap = torch.matmul(W,H) + eps_
    V_res = torch.div(V, V_ap)
    denom = torch.sum(H,1) + torch.div(phi*W,lambda_) + eps_
    update = torch.pow(torch.div(torch.matmul(V_res,H.transpose(0,1)),denom),0.5)
    return W * update

def update_W_gaussian_L1(H,W,lambda_,phi,V,eps_):
    #beta = 2 gamma(beta) = 1
    V_ap = torch.matmul(W,H).type(V.dtype) + eps_
    denom = torch.matmul(V_ap,H.transpose(0,1).type(V.dtype)) + torch.div(phi,lambda_).type(V.dtype) + eps_
    update = torch.div(torch.matmul(V,H.transpose(0,1).type(V.dtype)),denom)
    return W * update.type(torch.float32)

def update_W_gaussian_L2(H,W,lambda_,phi,V,eps_):
    #beta = 2 zeta(beta) = 1
    V_ap = torch.matmul(W,H) + eps_
    denom = torch.matmul(V_ap,H.transpose(0,1)) + torch.div(phi*W,lambda_) + eps_
    update = torch.div(torch.matmul(V,H.transpose(0,1)),denom)
    return W * update

# update tolerance value for early stop criteria
def update_del(lambda_, lambda_last):
    del_ = torch.max(torch.div(torch.abs(lambda_ - lambda_last)), lambda_last)
    return del_



def update_lambda_L1(W,H,b0,C,eps_):
    return torch.div(torch.sum(W,0) + torch.sum(H,1) + b0, C)

def update_lambda_L2(W,H,b0,C,eps_):
    return torch.div(0.5*torch.sum(W*W,0) + (0.5*torch.sum(H*H,1))+b0,C)

def update_lambda_L1_L2(W,H,b0,C,eps_):
    return torch.div(torch.sum(W,0) + 0.5*torch.sum(H*H,1)+b0,C)

def update_lambda_L2_L1(W,H,b0,C,eps_):
    return torch.div(0.5*torch.sum(torch.pow(W,2),0) + torch.sum(H,1)+b0,C)
