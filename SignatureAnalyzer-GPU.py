import pandas as pd
import numpy as np
import sys
import argparse
import time
from scipy.special import gamma
import os
import pickle
import torch
import NMF_functions
from ARD_NMF import ARD_NMF
import feather
from ARD_NMF import run_method_engine
import torch.nn as nn
import torch.multiprocessing as mp

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def run_parameter_sweep(parameters, data, args, Beta):
    output = []
    num_processes = torch.cuda.device_count()
    batches = int(len(parameters) / num_processes)
    idx = 0
    objectives = []
    nsigs = []
    while idx <= len(parameters)-num_processes:
        print(idx)
        pipe_list = []
        processes = []
        for rank in range(num_processes):
            recv_end, send_end = mp.Pipe(False)
            p = mp.Process(target=run_method_engine, args=(
                data,
                parameters.iloc[idx+rank]['a'],
                parameters.iloc[idx+rank]['phi'],
                parameters.iloc[idx+rank]['b'],
                Beta,
                args.prior_on_W,
                args.prior_on_H,
                parameters.iloc[idx+rank]['K0'],
                args.tolerance,
                args.max_iter,
                args.report_frequency,
                1e-5,
                send_end,
                rank,
                ))
            pipe_list.append(recv_end)
            processes.append(p)
            p.start()

        result_list = [x.recv() for x in pipe_list]
        for p in processes:
            p.join()

        nsig = [write_output(
            x[0],
            x[1],
            data.channel_names,
            data.sample_names,
            args.output_dir,
            parameters['label'][idx+i]
            ) for i,x in enumerate(result_list)]

        [nsigs.append(ns) for i,ns in enumerate(nsig)]
        [objectives.append(x[2]) for i,x in enumerate(result_list)]
        idx += num_processes

    if idx < len(parameters):
        for i in range(len(parameters)-idx):
            idx+=i
            x = run_method_engine(
                data,
                parameters.iloc[idx]['a'],
                parameters.iloc[idx]['phi'],
                parameters.iloc[idx]['b'],
                Beta,
                args.prior_on_W,
                args.prior_on_H,
                parameters.iloc[idx]['K0'],
                args.tolerance,
                args.max_iter,
                args.report_frequency,
                1e-5,
                send_end,
                rank
            )

            nsig = write_output(
                x[0],
                x[1],
                data.channel_names,
                data.sample_names,
                args.output_dir,
                parameters['label'][idx])

            nsigs.append(nsig)
            objectives.append(x[2])

    parameters['nsigs'] = nsigs
    parameters['objective'] = objectives
    parameters.to_csv(args.output_dir + '/parameters_with_results.txt',sep='\t',index=None)

def write_output(W, H, channel_names, sample_names, output_directory, label, active_thresh = 1e-5):
    createFolder(output_directory)
    nonzero_idx = (np.sum(H, axis=1) * np.sum(W, axis=0)) > active_thresh
    W_active = W[:, nonzero_idx]
    H_active = H[nonzero_idx, :]
    nsig = np.sum(nonzero_idx)
    # Normalize W and transfer weight to H matrix
    W_weight = np.sum(W_active, axis=0)
    W_final = W_active / W_weight
    H_final = W_weight[:, np.newaxis] * H_active

    sig_names = ['W' + str(j) for j in range(1, nsig + 1)]
    W_df = pd.DataFrame(data=W_final, index=channel_names, columns=sig_names)
    H_df = pd.DataFrame(data=H_final, index=sig_names, columns=sample_names);

    # Write W and H matrices
    W_df.to_csv(output_directory + '/'+label+ '_W.txt', sep='\t')
    H_df.to_csv(output_directory + '/'+label+ '_H.txt', sep='\t')

    return nsig

def main():
    ''' Run ARD NMF'''
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(
        description='NMF with some sparsity penalty described https://arxiv.org/pdf/1111.6085.pdf')
    parser.add_argument('--data', help='Data Matrix', required=True)
    parser.add_argument('--feather', help='Input in feather format', required=False, default=False, action='store_true')
    parser.add_argument('--parquet', help='Input in parquet format', required=False, default=False, action='store_true')
    parser.add_argument('--K0', help='Initial K parameter', required=False, default=None, type=int)
    parser.add_argument('--max_iter', help='maximum iterations', required=False, default=10000, type=int)
    parser.add_argument('--del_', help='Early stop condition based on lambda change', required=False, default=1,
                        type=int)
    parser.add_argument('--tolerance', help='Early stop condition based on max lambda entry', required=False, default=1e-6,
                        type=float)
    parser.add_argument('--phi', help='dispersion parameter see paper for discussion of choosing phi '
                                      'default = 1', required=False, default=1.0, type=float)
    parser.add_argument('--a', help='Hyperparamter for lambda. We recommend trying various values of a. Smaller values'
                                    'will result in sparser results a good starting point might be'
                                    'a = log(F+N)', required=False, default=10.0,type=float)

    parser.add_argument('--b', help='Hyperparamter for lambda. Default used is as recommended in Tan and Fevotte 2012',
                        required = False,type=float, default = None)
    parser.add_argument('--objective',help='Defines the data objective. Choose between "poisson" or "gaussian". Defaults to Poisson',
                        required=False,default='poisson',type=str)

    parser.add_argument('--prior_on_W',help = 'Prior on W matrix "L1" (exponential) or "L2" (half-normal)'
                        ,required = False, default = 'L1',type=str)
    parser.add_argument('--prior_on_H',help = 'Prior on H matrix "L1" (exponential) or "L2" (half-normal)'
                        ,required = False, default = 'L1',type=str)

    parser.add_argument('--output_dir', help='output_file_name if run in array mode this correspond to the output directory', required=True)
    parser.add_argument('--labeled', help='Input has row and column labels', required=False,default=False, action='store_true')
    parser.add_argument('--report_frequency', help='Number of iterations between progress reports', required=False,
                        default=100, type=int)
    parser.add_argument('--dtype', help='Floating point accuracy', required=False,
                        default='Float32', type=str)
    parser.add_argument('--parameters_file', help='allows running many different configurations of the NMF method on a multi'
                                                  'GPU system. To run in this mode provide this argument with a text file with '
                                                  'the following headers:(a,phi,b,prior_on_W,prior_on_H,Beta,label) label '
                                                  'indicates the output stem of the results from each run.', required = False
                                                  ,default = None)
    args = parser.parse_args()


    print('Reading data frame from '+ args.data)

    if args.dtype == 'Float32':
        args.dtype = torch.float32
    elif args.dtype == 'Float16':
        args.dtype = torch.float16

    if args.parquet:
        dataset = pd.read_parquet(args.data)
    elif args.feather:
        print('loading feather...')
        dataset = feather.read_dataframe(args.data)
    else:
        if args.labeled:
            dataset = pd.read_csv(args.data, sep='\t', header=0, index_col=0)
        else:
            dataset = pd.read_csv(args.data, sep='\t', header=None)


    if args.objective.lower() == 'poisson':
        Beta = 1
    elif args.objective.lower() == 'gaussian':
        Beta = 2
    else:
        print('objective parameter should be one of "gaussian" or "poisson"')
        sys.exit()

    data = ARD_NMF(dataset, args.objective)

    if args.parameters_file != None:
        parameters = pd.read_csv(args.parameters_file, sep='\t')
        run_parameter_sweep(parameters, data, args, Beta)
    else:
        x = run_method_engine(
            data,
            args.a,
            args.phi,
            args.b,
            Beta,
            args.prior_on_W,
            args.prior_on_H,
            args.K0,
            args.tolerance,
            args.max_iter,
            args.report_frequency,
        )
        nsig = write_output(x[0],x[1],data.channel_names,data.sample_names,args.output_dir,args.output_dir)

if __name__ == "__main__":
    main()
