import pandas as pd
import numpy as np
import sys
import argparse
import time
from scipy.special import gamma
import os
import pickle
import torch

class ARD_NMF:
    """
    NMF results class implements both half normal and exponential prior ARD NMF
    implementation based on https://arxiv.org/pdf/1111.6085.pdf
    """
    def __init__(self,dataset,objective,phi,a,b,K0 = None,prior_W = 'L1',prior_H = 'L2'):
        self.eps_ = torch.tensor(1.e-10,dtype=torch.float32,requires_grad=False)
        self.dataset = dataset
        self.V0 = torch.tensor(self.dataset.values[np.sum(self.dataset, axis=1) > 0, :]
                               ,dtype=torch.float32,requires_grad=False)
        self.V = self.V0 - torch.min(self.V0) + 1.e-30
        self.V_max = torch.max(self.V)
        self.M = self.V.shape[0]
        self.N = self.V.shape[1]
        if K0 == None:
            self.K0 = self.M
            self.number_of_active_components = self.M
        else:
            self.K0 = K0
            self.number_of_active_components = self.K0
        if objective.lower() == 'poisson':

            self.phi = torch.tensor(phi,dtype=torch.float32,requires_grad=False)

        else:
            self.phi = torch.tensor(phi,dtype=torch.float32,requires_grad=False)
            self.phi = torch.var(self.V)* self.phi

        self.a = a
        self.prior_W = prior_W
        self.prior_H = prior_H
        self.C = []
        self.b = b
        self.channel_names = self.dataset.index
        self.sample_names = self.dataset.columns
        print('NMF class initalized')

    def initalize_data(self):

        W0 = np.multiply(np.random.uniform(size=[self.M, self.K0]), np.sqrt(self.V_max.numpy()))
        H0 = np.multiply(np.random.uniform(size=[self.K0, self.N]), np.sqrt(self.V_max.numpy()))
        L0 = np.sum(W0,axis=0) + np.sum(H0,axis=1)

        self.W = torch.tensor(W0, dtype=torch.float32, requires_grad=False)
        self.H = torch.tensor(H0, dtype=torch.float32, requires_grad=False)
        self.Lambda = torch.tensor(L0, dtype=torch.float32, requires_grad=False)


        # calculate default b as described in Tan and Fevotte (2012)
        if self.b == None or self.b == 'None':
            # L1 ARD
            if self.prior_H == 'L1' and self.prior_W == 'L1':
                
                self.bcpu = np.sqrt(np.true_divide( (self.a - 1)*(self.a - 2) * torch.mean(self.V).numpy(),self.K0 ))
                self.b = torch.tensor(
                    np.sqrt(np.true_divide( (self.a - 1)*(self.a - 2) * torch.mean(self.V).numpy(),self.K0 ))
                    ,dtype=torch.float32,requires_grad=False)
                
                self.C = torch.tensor(self.N + self.M + self.a + 1, dtype=torch.float32, requires_grad=False)
            # L2 ARD
            elif self.prior_H == 'L2' and self.prior_W == 'L2':
                
                self.bcpu = np.true_divide(np.pi * (self.a - 1) * torch.mean(self.V).numpy(),2*self.K0)
                self.b = torch.tensor(
                    np.true_divide(np.pi * (self.a - 1) * torch.mean(self.V).numpy(),2*self.K0),
                    dtype=torch.float32,requires_grad=False)
                
                self.C = torch.tensor( (self.N + self.M)*0.5 + self.a + 1, dtype=torch.float32,requires_grad=False)
                
            # L1 - L2 ARD
            elif self.prior_H == 'L1' and self.prior_W == 'L2':
                self.bcpu = np.true_divide(torch.mean(self.V).numpy()*np.sqrt(2)*gamma(self.a-3/2),self.K0*np.sqrt(np.pi)*gamma(self.a))
                self.b = torch.tensor(
                    np.true_divide(torch.mean(self.V).numpy()*np.sqrt(2)*gamma(self.a-3/2),self.K0*np.sqrt(np.pi)*gamma(self.a))
                    ,dtype=torch.float32,requires_grad=False)
                self.C = torch.tensor(self.N + self.M/2 + self.a + 1, dtype=torch.float32)
            elif self.prior_H == 'L2' and self.prior_W == 'L1':
                self.bcpu = np.true_divide(torch.mean(self.V).numpy()*np.sqrt(2)*gamma(self.a-3/2),self.K0*np.sqrt(np.pi)*gamma(self.a))
                self.b = torch.tensor(
                    np.true_divide(torch.mean(self.V).numpy()*np.sqrt(2)*gamma(self.a-3/2),self.K0*np.sqrt(np.pi)*gamma(self.a)),
                    dtype=torch.float32,requires_grad=False)
                self.C = torch.tensor(self.N/2 + self.M + self.a + 1, dtype=torch.float32)
        else:
            self.bcpu = self.b
            self.b = torch.tensor(self.b, dtype=torch.float32,requires_grad=False)
            if self.prior_H == 'L1' and self.prior_W == 'L1':
                self.C = torch.tensor(self.N + self.M + self.a + 1, dtype=torch.float32,requires_grad=False)
            # L2 ARD
            elif self.prior_H == 'L2' and self.prior_W == 'L2':
                self.C = torch.tensor( (self.N + self.M)*0.5 + self.a + 1, dtype=torch.float32,requires_grad=False)
            # L1 - L2 ARD
            elif self.prior_H == 'L1' and self.prior_W == 'L2':
                self.C = torch.tensor(self.N + self.M/2 + self.a + 1, dtype=torch.float32,requires_grad=False)
            elif self.prior_H == 'L2' and self.prior_W == 'L1':
                self.C = torch.constant(self.N/2 + self.M + self.a + 1, dtype=torch.float32,requires_grad=False)


    def get_number_of_active_components(self):
        self.number_of_active_components = torch.sum(torch.sum(self.W,0)> 0.0, dtype=torch.float32)