import pandas as pd
import numpy as np
import sys
import argparse
import time
from scipy.special import gamma
import os
import pickle
import torch
from NMF_functions import *

class ARD_NMF:
    """
    NMF results class implements both half normal and exponential prior ARD NMF
    implementation based on https://arxiv.org/pdf/1111.6085.pdf
    """
    def __init__(self,dataset,objective,dtype = torch.float32):
        self.eps_ = torch.tensor(1.e-7,dtype=dtype,requires_grad=False)
        self.dataset = dataset
        zero_idx = np.sum(self.dataset, axis=1) > 0
        self.V0 = self.dataset.values[zero_idx, :]
        self.V = self.V0 - np.min(self.V0) + 1.e-30
        self.V_max = np.max(self.V)
        self.M = self.V.shape[0]
        self.N = self.V.shape[1]
        self.objective = objective
        self.channel_names = self.dataset.index[zero_idx]
        self.sample_names = self.dataset.columns
        self.dtype = dtype
        print('NMF class initalized.')

    def initalize_data(self,a,phi,b,prior_W,prior_H,Beta,K0,dtype = torch.float32):
        
        
        if K0 == None:
            self.K0 = self.M
            self.number_of_active_components = self.M
        else:
            self.K0 = K0
            self.number_of_active_components = self.K0
            
        if self.objective.lower() == 'poisson':

            self.phi = torch.tensor(phi,dtype=dtype,requires_grad=False)

        else:
            self.phi = torch.tensor(np.var(self.V)* phi,dtype=dtype,requires_grad=False)
        
        self.a = a
        self.prior_W = prior_W
        self.prior_H = prior_H
        self.C = []
        self.b = b
        
        
        W0 = np.multiply(np.random.uniform(size=[self.M, self.K0])+self.eps_.numpy(), np.sqrt(self.V_max))
        H0 = np.multiply(np.random.uniform(size=[self.K0, self.N])+self.eps_.numpy(), np.sqrt(self.V_max))
        L0 = np.sum(W0,axis=0) + np.sum(H0,axis=1)

        self.W = torch.tensor(W0, dtype=self.dtype, requires_grad=False)
        self.H = torch.tensor(H0, dtype=self.dtype, requires_grad=False)
        self.Lambda = torch.tensor(L0, dtype=torch.float32, requires_grad=False)


        # calculate default b as described in Tan and Fevotte (2012)
        if self.b == None or self.b == 'None':
            # L1 ARD
            if self.prior_H == 'L1' and self.prior_W == 'L1':
                
                self.bcpu = np.sqrt(np.true_divide( (self.a - 1)*(self.a - 2) * np.mean(self.V),self.K0 ))
                self.b = torch.tensor(
                    np.sqrt(np.true_divide( (self.a - 1)*(self.a - 2) * np.mean(self.V),self.K0 ))
                    ,dtype=self.dtype,requires_grad=False)
                
                self.C = torch.tensor(self.N + self.M + self.a + 1, dtype=self.dtype, requires_grad=False)
            # L2 ARD
            elif self.prior_H == 'L2' and self.prior_W == 'L2':
                
                self.bcpu = np.true_divide(np.pi * (self.a - 1) * np.mean(self.V),2*self.K0)
                self.b = torch.tensor(
                    np.true_divide(np.pi * (self.a - 1) * np.mean(self.V),2*self.K0),
                    dtype=self.dtype,requires_grad=False)
                
                self.C = torch.tensor( (self.N + self.M)*0.5 + self.a + 1, dtype=self.dtype,requires_grad=False)
                
            # L1 - L2 ARD
            elif self.prior_H == 'L1' and self.prior_W == 'L2':
                self.bcpu = np.true_divide(np.mean(self.V)*np.sqrt(2)*gamma(self.a-3/2),self.K0*np.sqrt(np.pi)*gamma(self.a))
                self.b = torch.tensor(
                    np.true_divide(np.mean(self.V)*np.sqrt(2)*gamma(self.a-3/2),self.K0*np.sqrt(np.pi)*gamma(self.a))
                    ,dtype=self.dtype,requires_grad=False)
                self.C = torch.tensor(self.N + self.M/2 + self.a + 1, dtype=self.dtype)
            elif self.prior_H == 'L2' and self.prior_W == 'L1':
                self.bcpu = np.true_divide(np.mean(self.V)*np.sqrt(2)*gamma(self.a-3/2),self.K0*np.sqrt(np.pi)*gamma(self.a))
                self.b = torch.tensor(
                    np.true_divide(np.mean(self.V)*np.sqrt(2)*gamma(self.a-3/2),self.K0*np.sqrt(np.pi)*gamma(self.a)),
                    dtype=self.dtype,requires_grad=False)
                self.C = torch.tensor(self.N/2 + self.M + self.a + 1, dtype=self.dtype)
        else:
            self.bcpu = self.b
            self.b = torch.tensor(self.b, dtype=self.dtype,requires_grad=False)
            if self.prior_H == 'L1' and self.prior_W == 'L1':
                self.C = torch.tensor(self.N + self.M + self.a + 1, dtype=self.dtype,requires_grad=False)
            # L2 ARD
            elif self.prior_H == 'L2' and self.prior_W == 'L2':
                self.C = torch.tensor( (self.N + self.M)*0.5 + self.a + 1, dtype=self.dtype,requires_grad=False)
            # L1 - L2 ARD
            elif self.prior_H == 'L1' and self.prior_W == 'L2':
                self.C = torch.tensor(self.N + self.M/2 + self.a + 1, dtype=self.dtype,requires_grad=False)
            elif self.prior_H == 'L2' and self.prior_W == 'L1':
                self.C = torch.constant(self.N/2 + self.M + self.a + 1, dtype=self.dtype,requires_grad=False)

        self.V = torch.tensor(self.V,dtype=self.dtype,requires_grad=False)
        print('NMF data and parameters set.')
    def get_number_of_active_components(self):
        self.number_of_active_components = torch.sum(torch.sum(self.W,0)> 0.0, dtype=self.dtype)
        
        
        
def run_method_engine(results, a, phi, b, Beta, W_prior, H_prior, K0, tolerance, max_iter, send_end = None, cuda_int = 0):
    # initalize the NMF run
    results.initalize_data(a,phi,b,W_prior,H_prior,Beta,K0)
    # specify GPU
    cuda_string = 'cuda:'+str(cuda_int)
    # copy data to GPU
    W,H,V,Lambda,C,b0,eps_,phi = results.W.cuda(cuda_string),results.H.cuda(cuda_string),results.V.cuda(cuda_string),results.Lambda.cuda(cuda_string),results.C.cuda(cuda_string),results.b.cuda(cuda_string),results.eps_.cuda(cuda_string),results.phi.cuda(cuda_string)    
    
    # tracking variables
    deltrack = 1000
    times = list()
    active_thresh = 1e-5
    iter = 0
    report_freq = 10
    lam_previous = Lambda
    print('%%%%%%%%%%%%%%%')
    print('a =',results.a)
    print('b =',results.bcpu)
    print('%%%%%%%%%%%%%%%')
    # set method
    method = NMF_algorithim(Beta,H_prior,W_prior)

    
    while deltrack >= tolerance and iter < max_iter:
        # compute updates
        H,W,Lambda = method.forward(W,H,V,Lambda,C,b0,eps_,phi)
        # compute objective and cost
        l_ = beta_div(Beta,V,W,H,eps_)
        cost_ = calculate_objective_function(Beta,V,W,H,Lambda,C,eps_,phi,results.K0)
        # update tracking
        deltrack = torch.max(torch.div(torch.abs(Lambda -lam_previous), (lam_previous+1e-5)))
        lam_previous = Lambda
        
        # report to stdout
        if iter % report_freq == 0:
            print("nit=%s\tobjective=%s\tbeta_div=%s\tlambda=%s\tdel=%s\tK=%s\tsumW=%s\tsumH=%s" % (iter,cost_.cpu().numpy(),l_.cpu().numpy(),torch.sum(Lambda).cpu().numpy()
                                                                                                ,deltrack.cpu().numpy(),
                                                                                                torch.sum((torch.sum(H,1) + torch.sum(W,0))>active_thresh).cpu().numpy()
                                                                                                ,torch.sum(W).cpu().numpy(),torch.sum(H).cpu().numpy()))
    
        iter+=1
    if send_end != None:    
        send_end.send([W.cpu().numpy(),H.cpu().numpy(),cost_.cpu().numpy()])
    else:
        return W.cpu().numpy(),H.cpu().numpy(),cost_.cpu().numpy()
        