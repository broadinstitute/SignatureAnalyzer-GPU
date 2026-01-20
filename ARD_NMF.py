import pandas as pd
import numpy as np
import sys
from sys import stdout
import argparse
import time
from scipy.special import gamma
import os
import pickle
import math
import torch
from typing import Union
import multiprocessing.connection as mpc
from .NMF_functions import *

class ARD_NMF:
    """
    NMF results class implements both half normal and exponential prior ARD NMF
    implementation based on https://arxiv.org/pdf/1111.6085.pdf
    """
    def __init__(self,dataset,objective,dtype = torch.float32, verbose=True):
        self.eps_ = torch.tensor(1.e-30,dtype=dtype,requires_grad=False)
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
        self.verbose = verbose
        if self.verbose: print('NMF class initalized.')

    def initalize_data(self,a,phi,b,prior_W,prior_H,Beta,K0,dtype = torch.float32):
        """
        Initializes dataset.

        Args:
            * a
            * phi: dispersion parameter - multiplied by variance if objective
                function is Gaussian (see Tan & Fevotte 2013)
            * b
            * prior_W
            * prior_H
            * Beta
            * K0: set to number of input features if not provided
        """
        print('NMF class initialized.')

    def initalize_data(self,a,phi,b,prior_W,prior_H,Beta,K0,use_val_set,dtype = torch.float32):
        
        self.V = np.array(self.V) #when gets called in a loop as in run_parameter_sweep this can get updated to a torch tensor in a previous iteration which breaks some numpy functions

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

        if use_val_set:
            torch.manual_seed(0) #get the same mask each time
            self.mask = (torch.rand(self.V.shape) > 0.2).type(self.dtype) #create mask, randomly mask ~20% of data in shape V. Only used when passed
        else:
            self.mask = torch.ones(self.V.shape, dtype=self.dtype)

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
                self.bcpu = np.true_divide(np.mean(self.V)*np.sqrt(np.pi)*gamma(self.a),self.K0*np.sqrt(2)*gamma(self.a-3/2))**(2/3)
                self.b = torch.tensor(
                    np.true_divide(np.mean(self.V)*np.sqrt(np.pi)*gamma(self.a),self.K0*np.sqrt(2)*gamma(self.a-3/2))**(2/3),
                    dtype=self.dtype,requires_grad=False)
                self.C = torch.tensor(self.N + self.M/2 + self.a + 1, dtype=self.dtype)
            elif self.prior_H == 'L2' and self.prior_W == 'L1':
                self.bcpu = np.true_divide(np.mean(self.V)*np.sqrt(np.pi)*gamma(self.a),self.K0*np.sqrt(2)*gamma(self.a-3/2))**(2/3)
                self.b = torch.tensor(
                    np.true_divide(np.mean(self.V)*np.sqrt(np.pi)*gamma(self.a),self.K0*np.sqrt(2)*gamma(self.a-3/2))**(2/3),
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
                self.C = torch.tensor(self.N/2 + self.M + self.a + 1, dtype=self.dtype,requires_grad=False)

        self.V = torch.tensor(self.V,dtype=self.dtype,requires_grad=False)
        if self.verbose: print('NMF data and parameters set.')

    def get_number_of_active_components(self):
        self.number_of_active_components = torch.sum(torch.sum(self.W,0)> 0.0, dtype=self.dtype)

def print_report(iter,report,verbose,tag):
    """
    Prints report.
    """
    if verbose:
        print("nit={:>5} K={:>5} | obj={:.2f}\tb_div={:.2f}\tlam={:.2f}\tdel={:.8f}\tsumW={:.2f}\tsumH={:.2f}".format(
            iter,
            report[iter]['K'],
            report[iter]['obj'],
            report[iter]['b_div'],
            report[iter]['lam'],
            report[iter]['del'],
            report[iter]['W_sum'],
            report[iter]['H_sum']
            )
        )
    else:
        stdout.write("\r{}nit={:>5} K={} \tdel={:.8f}".format(
            tag,
            iter,
            report[iter]['K'],
            report[iter]['del']
            )
        )

def run_method_engine(
    results: ARD_NMF,
    a: float,
    phi: float,
    b: float,
    Beta: int,
    W_prior: str,
    H_prior: str,
    K0: int,
    tolerance: float,
    max_iter: int,
    use_val_set: bool,
    report_freq: int = 10,
    active_thresh: float = 1e-5,
    send_end: Union[mpc.Connection, None] = None,
    cuda_int: Union[int, None] = 0,
    verbose: bool = True,
    tag: str = ""
    ) -> (pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray):
    """
    Run ARD-NMF Engine.
    ------------------------------------------------------------------------
    Args:
        * results: initialized ARD_NMF class
        * a: shape parameter
        * phi: dispersion parameter
        * b: shape parameter
        * Beta: defined by objective function
        * W_prior: prior on W matrix ("L1" or "L2")
        * H_prior: prior on H matrix ("L1" or "L2")
        * K0: starting number of latent components
        * tolerance: end-point of optimization
        * max_iter: maximum number of iterations for algorithm
        * use_val_set: use validation set for ARD-NMF
          If False (default), set masks to all ones. 
          Otherwise, use 0/1 mask to hold out 0's as validation set during training and will report objective function value for that set.
        * report_freq: how often to print updates
        * active_thresh: threshold for a latent component's impact on
            signature if the latent factor is less than this, it does not contribute
        * send_end: mpc.Connection resulting from multiprocessing.Pipe,
            for use in parameter sweep implementation
        * cuda_int: GPU to use. Defaults to 0. If "None" or if no GPU available,
            will perform decomposition using CPU.
        * verbose: verbose logging

    Returns:
        * H: (samples x K)
        * W: (K x features)
        * markers
        * signatures
    """
    # initalize the NMF run
    results.initalize_data(a, phi, b, W_prior, H_prior, Beta, K0, use_val_set)
    # specify GPU
    cuda_string = 'cuda:'+str(cuda_int)
    # copy data to GPU
    if torch.cuda.device_count() > 0 and cuda_int is not None:
        if verbose: print("   * Using GPU: {}".format(cuda_string))
        W,H,V,Lambda,C,b0,eps_,phi,mask = results.W.cuda(cuda_string),results.H.cuda(cuda_string),results.V.cuda(cuda_string),results.Lambda.cuda(cuda_string),results.C.cuda(cuda_string),results.b.cuda(cuda_string),results.eps_.cuda(cuda_string),results.phi.cuda(cuda_string),results.mask.cuda(cuda_string)
    else:
        W,H,V,Lambda,C,b0,eps_,phi,mask = results.W,results.H,results.V,results.Lambda,results.C,results.b,results.eps_,results.phi,results.mask
        if verbose: print("   * Using CPU")

    # tracking variables
    deltrack = 1000
    times = list()
    report = dict()
    iter = 0
    lam_previous = Lambda
    if verbose: print('%%%%%%%%%%%%%%%')
    if verbose: print('a =',results.a)
    if verbose: print('b =',results.bcpu)
    if verbose: print('%%%%%%%%%%%%%%%')

    # set method
    method = NMF_algorithim(Beta, H_prior, W_prior)

    start_time = time.time()
    while deltrack >= tolerance and iter < max_iter:
        # compute updates
        H,W,Lambda = method.forward(W,H,V,Lambda,C,b0,eps_,phi,mask)

        # compute objective and cost (excluding validation set, when mask is passed)
        l_ = beta_div(Beta,V,W,H,eps_,mask)
        cost_ = calculate_objective_function(Beta,V,W,H,Lambda,C,eps_,phi,results.K0,mask)

        # update tracking
        deltrack = torch.max(torch.div(torch.abs(Lambda-lam_previous), lam_previous+1e-30))
        lam_previous = Lambda

        # ---------------------------- Reporting ---------------------------- #
        if iter % report_freq == 0:
            report[iter] = {
                'K': torch.sum((torch.sum(H,1) * torch.sum(W,0))>active_thresh).cpu().numpy(),
                'obj': cost_.cpu().numpy(),
                'b_div': l_.cpu().numpy(),
                'lam': torch.sum(Lambda).cpu().numpy(),
                'del': deltrack.cpu().numpy(),
                'W_sum': torch.sum(W).cpu().numpy(),
                'H_sum': torch.sum(H).cpu().numpy()
            }
            print_report(iter,report,verbose,tag)
        # ------------------------------------------------------------------- #
        iter+=1


    # --------------------------- Final Report --------------------------- #
    report[iter] = {
        'K': torch.sum((torch.sum(H,1) * torch.sum(W,0))>active_thresh).cpu().numpy(),
        'obj': cost_.cpu().numpy(),
        'b_div': l_.cpu().numpy(),
        'lam': torch.sum(Lambda).cpu().numpy(),
        'del': deltrack.cpu().numpy(),
        'W_sum': torch.sum(W).cpu().numpy(),
        'H_sum': torch.sum(H).cpu().numpy()
    }
    
    end_time = time.time()
    
    #compute validation set performance
    if use_val_set: 
        heldout_mask = 1-mask #now select heldout values (inverse of mask)
        report[iter]['b_div_val'] = beta_div(Beta,V,W,H,eps_,heldout_mask)
        report[iter]['obj_val'] = calculate_objective_function(Beta,V,W,H,Lambda,C,eps_,phi,results.K0,heldout_mask)
        #print("validation set objective=%s\tbeta_div=%s" % (cost_.cpu().numpy(),l_.cpu().numpy()))
    else:
        report[iter]['b_div_val'] = None
        report[iter]['obj_val'] = None
        
    print_report(iter,report,verbose,tag)

    if not verbose:
        stdout.write("\n")
    # ------------------------------------------------------------------- #

    if send_end != None:
        send_end.send([W.cpu().numpy(), H.cpu().numpy(), mask.cpu().numpy(), cost_.cpu().numpy(), l_.cpu().numpy(), report[iter]['b_div_val'].cpu().numpy(), report[iter]['obj_val'].cpu().numpy(), end_time-start_time,])
    else:
        final_report = pd.DataFrame.from_dict(report).T
        final_report.index.name = 'iter'
        return W.cpu().numpy(), H.cpu().numpy(), cost_.cpu().numpy(), final_report, Lambda.cpu().numpy(), mask.cpu().numpy()