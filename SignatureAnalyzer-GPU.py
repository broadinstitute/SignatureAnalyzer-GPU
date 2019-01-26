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


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)




def get_available_gpus():
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    local_device_protos = device_lib.list_local_devices()
    sess.close()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def apply_array_updates(sess,updates,gen_array,prime_array):
    for i in range(len(updates)):
        sess.run(gen_array[i], feed_dict={prime_array[i]: updates[i]})

def run_NMF_parameter_search(parameters,data,objective,max_iter=10000,report_freq=100,tol_=1e-5,active_thresh=1e-5,output_directory='.'):
    # parameters in terms of pandas data frame with columns (a,phi,b,prior_on_W,prior_on_H,Beta,label)
    # will return a dictionary of {label : NMF_results_object}
    GPUs = get_available_gpus()
    n_GPUs = len(GPUs)
    parameter_batches = int(np.ceil(np.true_divide(len(parameters), n_GPUs)))
    job_dict = dict()
    parameter_index = 0
    objectives = list()
    n_active = list()
    job_counter = 0
    result_index = 0
    for batch in range(0,parameter_batches):
        labels = []
        h_array = list()
        w_array = list()
        lambda_array = list()
        beta_div_array = list()
        objective_array = list()
        updates_gen_H = list()
        updates_prime_H = list()
        updates_gen_W = list()
        updates_prime_W = list()
        updates_gen_Lambda = list()
        updates_prime_Lambda = list()
        lam_previous_array = list()
        for G in GPUs:
            if job_counter >= len(parameters):
                ## handling last batch size < n_gpus
                break
            r = parameters.iloc[job_counter]
            job_counter+=1
            print('Running job '+r['label'])
            if objective == None:
                if r['Beta'] == 1:
                    objective = 'poisson'
                elif r['Beta'] == 2:
                    objective = 'gaussian'
                else:
                    print('ERROR: One of Beta and/or objective are required to be defined.')
                    sys.exit()
            job_dict[r['label']] = ARD_NMF(data, objective,
                                           r['phi'], r['a'], r['b'], r['K0'], r['prior_on_W'], r['prior_on_H'])
            job_dict[r['label']].initalize_data()
            labels.append(r['label'])
            if parameter_index <= len(parameters):
             with tf.device(G):
                print('%%%%%%%%%%%%%%%')
                print('a =', job_dict[parameters['label'][parameter_index]].a)
                print('b =', job_dict[parameters['label'][parameter_index]].bcpu)
                print('%%%%%%%%%%%%%%%')
                method = NMF_functions.NMF_algorithim(Beta=parameters['Beta'][parameter_index]
                                                  ,H_prior=parameters['prior_on_H'][parameter_index]
                                                  ,W_prior=parameters['prior_on_W'][parameter_index])

    
                h_ = method.update_H(job_dict[parameters['label'][parameter_index]].H,
                                 job_dict[parameters['label'][parameter_index]].W,
                                 job_dict[parameters['label'][parameter_index]].Lambda,
                                 job_dict[parameters['label'][parameter_index]].phi,
                                 job_dict[parameters['label'][parameter_index]].V,
                                 job_dict[parameters['label'][parameter_index]].eps_)

                w_ = method.update_W(job_dict[parameters['label'][parameter_index]].H,
                                 job_dict[parameters['label'][parameter_index]].W,
                                 job_dict[parameters['label'][parameter_index]].Lambda,
                                 job_dict[parameters['label'][parameter_index]].phi,
                                 job_dict[parameters['label'][parameter_index]].V,
                                 job_dict[parameters['label'][parameter_index]].eps_)

                lambda_ = method.lambda_update(job_dict[parameters['label'][parameter_index]].W,
                                        job_dict[parameters['label'][parameter_index]].H,
                                        job_dict[parameters['label'][parameter_index]].b,
                                        job_dict[parameters['label'][parameter_index]].C,
                                        job_dict[parameters['label'][parameter_index]].eps_)

                gen_H, h_prime = NMF_functions.assign_updates(job_dict[parameters['label'][parameter_index]].H)
                gen_W, w_prime = NMF_functions.assign_updates(job_dict[parameters['label'][parameter_index]].W)
                gen_Lambda, lambda_prime = NMF_functions.assign_updates(job_dict[parameters['label'][parameter_index]].Lambda)


                l_ = NMF_functions.beta_div(parameters['Beta'][parameter_index],
                                        job_dict[parameters['label'][parameter_index]].V,
                                        job_dict[parameters['label'][parameter_index]].W,
                                        job_dict[parameters['label'][parameter_index]].H,
                                        job_dict[parameters['label'][parameter_index]].eps_)

                cost_ = NMF_functions.calculate_objective_function(parameters['Beta'][parameter_index],
                                                               job_dict[parameters['label'][parameter_index]].V,
                                                               job_dict[parameters['label'][parameter_index]].W,
                                                               job_dict[parameters['label'][parameter_index]].H,
                                                               job_dict[parameters['label'][parameter_index]].Lambda,
                                                               job_dict[parameters['label'][parameter_index]].C,
                                                               job_dict[parameters['label'][parameter_index]].eps_,
                                                               job_dict[parameters['label'][parameter_index]].phi,
                                                               job_dict[parameters['label'][parameter_index]].K0)
                lam_previous = 1
                parameter_index += 1
                h_array.append(h_)
                w_array.append(w_)
                lambda_array.append(lambda_)
                beta_div_array.append(l_)
                objective_array.append(cost_)
                updates_gen_H.append(gen_H)
                updates_prime_H.append(h_prime)
                updates_gen_W.append(gen_W)
                updates_prime_W.append(w_prime)
                updates_gen_Lambda.append(gen_Lambda)
                updates_prime_Lambda.append(lambda_prime)
                lam_previous_array.append(lam_previous)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        iter = 0
        deltrack = np.ones([len(h_array),1])

        with tf.Session() as sess:
            sess.run(init_op)

            while np.max(deltrack) >= tol_ and iter < max_iter:

                h_new = sess.run(h_array)
                apply_array_updates(sess,updates=h_new,gen_array=updates_gen_H,prime_array=updates_prime_H)

                w_new = sess.run(w_array)
                apply_array_updates(sess, updates=w_new, gen_array=updates_gen_W, prime_array=updates_prime_W)

                lambda_new = sess.run(lambda_array)
                apply_array_updates(sess, updates=lambda_new, gen_array=updates_gen_Lambda, prime_array=updates_prime_Lambda)

                if iter % report_freq == 0:
                    l_new = sess.run(beta_div_array)
                    cost_new = sess.run(objective_array)
                    for i in range(len(h_new)):
                        print("nit=%s\tobjective=%s\tbeta_div=%s\tlambda=%s\tdel=%s\tK=%s\tsumW=%s\tsumH=%s" % (
                        iter, cost_new[i], l_new[i], np.sum(lambda_new[i]), deltrack[i],
                        np.array2string(np.sum((np.sum(h_new[i], axis=1) * np.sum(w_new[i], axis=0)) > active_thresh)),
                        np.sum(w_new[i]), np.sum(h_new[i])))
                iter = iter + 1
                for i in range(len(h_new)):
                    deltrack[i] = np.max(np.true_divide(np.abs(lambda_new[i] - lam_previous_array[i]), (lam_previous_array[i] + 1e-5)))
                    lam_previous_array[i] = lambda_new[i]


        for i in range(len(h_new)):
            nonzero_idx = (np.sum(h_new[i], axis=1) * np.sum(w_new[i], axis=0)) > active_thresh
            W_active = w_new[i][:, nonzero_idx]
            H_active = h_new[i][nonzero_idx, :]
            Lambda_k = lambda_new[i][nonzero_idx]
            nsig = np.sum(nonzero_idx)
            # Normalize W and transfer weight to H matrix
            W_weight = np.sum(W_active, axis=0)
            W_final = W_active / W_weight
            H_final = W_weight[:, np.newaxis] * H_active

            sig_names = ['W' + str(j) for j in range(1, nsig + 1)]
            W_df = pd.DataFrame(data=W_final, index=job_dict[parameters['label'][result_index]].channel_names, columns=sig_names)
            H_df = pd.DataFrame(data=H_final, index=sig_names, columns=job_dict[parameters['label'][result_index]].sample_names);

            # Write W and H matrices
            W_df.to_csv(output_directory + '/'+parameters['label'][result_index]+ '_W.txt', sep='\t')
            H_df.to_csv(output_directory + '/'+parameters['label'][result_index]+ '_H.txt', sep='\t')

            n_active.append(nsig)
            file = open(output_directory + '/'+parameters['label'][result_index]+ '_n_signatures.txt', 'w')
            file.write('%s\n' % (np.array2string(nsig)))
            file.close()

            objectives.append(cost_new[i])
            file = open(output_directory + '/'+parameters['label'][result_index]+  '_objective_function.txt', 'w')
            file.write('%s\n' % (np.array2string(cost_new[i])))
            file.close()

            with open(output_directory + '/'+parameters['label'][result_index]+  '_results.pkl', 'wb') as f:
                pickle.dump([W_active,H_active,Lambda_k], f)
            result_index += 1
        for label in labels:
            job_dict[label] = []

    parameters['objective'] = objectives
    parameters['n_active'] = nsig
    parameters.to_csv(output_directory + '/parameters_with_results.txt',sep='\t',index=None)

def main():
    ''' Run ARD NMF'''

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

    parser.add_argument('--output_file', help='output_file_name if run in array mode this correspond to the output directory', required=True)
    parser.add_argument('--labeled', help='Input has row and column labels', required=False,default=False, action='store_true')
    parser.add_argument('--report_frequency', help='Number of iterations between progress reports', required=False,
                        default=100, type=int)

    parser.add_argument('--parameters_file', help='allows running many different configurations of the NMF method on a multi'
                                                  'GPU system. To run in this mode provide this argument with a text file with '
                                                  'the following headers:(a,phi,b,prior_on_W,prior_on_H,Beta,label) label '
                                                  'indicates the output stem of the results from each run.', required = False
                                                    ,default = None)
    args = parser.parse_args()


    print('Reading data frame from '+ args.data)


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
        #if any(dataset.sum(1) > 0):
        #    dataset = dataset[dataset.sum(1) > 0]
        #    print('WARNING: Dropping rows with zero sum')

    if args.parameters_file == None:
        if args.objective.lower() == 'poisson':
            Beta = 1
        elif args.objective.lower() == 'gaussian':
            Beta = 2
        else:
            print('objective parameter should be one of "gaussian" or "poisson"')
            sys.exit()

        max_iter = args.max_iter
        del_ = args.del_
        tol_ = args.tolerance
        n_lambda = []

        # create new results object containing H W and V
        results = ARD_NMF(dataset,args.objective,args.phi,args.a,args.b,args.K0,args.prior_on_W,args.prior_on_H)
        #
        results.initalize_data()

        print('%%%%%%%%%%%%%%%')
        print('a =',results.a)
        print('b =',results.bcpu)
        print('%%%%%%%%%%%%%%%')

        results.W,results.H,results.V,results.Lambda,results.C,results.b0,results.eps_,results.phi = results.W.type(torch.float16).cuda(),results.H.type(torch.float16).cuda(),results.V.type(torch.float16).cuda(),results.Lambda.type(torch.float16).cuda(),results.C.type(torch.float16).cuda(),results.b.type(torch.float16).cuda(),results.eps_.type(torch.float16).cuda(),results.phi.type(torch.float16).cuda()
        method = NMF_functions.NMF_algorithim(Beta=Beta,H_prior=args.prior_on_H,W_prior=args.prior_on_W)
        # Threshhold for signature to be considered "active"
        active_thresh = 1e-5

        # How many iterations between printing progress updates
        deltrack = 1000
        times = list()
        active_thresh = 1e-5
        iter = 0
        report_freq = args.report_frequency
        lam_previous = results.Lambda
        times = list()
        while deltrack >= tol_ and iter < max_iter:
            start = time.time()
            results.H,results.W,results.Lambda = method.algorithm(results.W,results.H,results.V,
                                                                  results.Lambda, results.C, results.b0,
                                                                  results.eps_,results.phi)
            print(results.H.cpu().numpy())
            print(results.W.cpu().numpy())
            print(results.Lambda.cpu().numpy())
            l_ = NMF_functions.beta_div(Beta,results.V,results.W,results.H,results.eps_)
            cost_ = NMF_functions.calculate_objective_function(Beta,results.V,results.W,results.H,results.Lambda,results.C,
                                                       results.eps_,results.phi,results.K0)
            deltrack = torch.max(torch.div(torch.abs(results.Lambda -lam_previous), (lam_previous+1e-5)))
            lam_previous = results.Lambda
            end = time.time()
            times.append(end-start)
            if iter % report_freq == 0:
                print("nit=%s\tobjective=%s\tbeta_div=%s\tlambda=%s\tdel=%s\tK=%s\tsumW=%s\tsumH=%s" % (iter,cost_.cpu().numpy(),l_.cpu().numpy(),torch.sum(results.Lambda).cpu().numpy()
                                                                                                ,deltrack.cpu().numpy(),
                                                                                                torch.sum((torch.sum(results.H,1) + torch.sum(results.W,0))>active_thresh).cpu().numpy()
                                                                                                ,torch.sum(results.W).cpu().numpy(),torch.sum(results.H).cpu().numpy()))
            
            iter+=1

        nonzero_idx = (torch.sum(results.H,1) + torch.sum(results.W, 0)) > active_thresh
        W_active = results.W[:, nonzero_idx].cpu().numpy()
        H_active = results.H[nonzero_idx, :].cpu().numpy()
        nsig = torch.sum(nonzero_idx).cpu().numpy()
        # Normalize W and transfer weight to H matrix
        W_weight = np.sum(W_active, axis=0)
        W_final = W_active / W_weight
        H_final = W_weight[:, np.newaxis] * H_active

        sig_names = ['W' + str(i) for i in range(1,nsig+1)]

        W_df = pd.DataFrame(data=W_final,index=results.channel_names,columns=sig_names)
        H_df = pd.DataFrame(data=H_final,index=sig_names,columns=results.sample_names);

        # Write W and H matrices
        W_df.to_csv(args.output_file + '_W.txt',sep='\t')
        H_df.to_csv(args.output_file + '_H.txt',sep='\t')

        file = open(args.output_file + '_n_signatures.txt', 'w')
        file.write('%s\n' % (np.array2string(nsig)))
        file.close()

        file = open(args.output_file + '_objective_function.txt', 'w')
        file.write('%s\n' % (np.array2string(cost_.cpu().numpy())))
        file.close()
        
        with open('times.txt', 'w') as f:
            for item in times:
                f.write("%s\n" % item)
        
    else:
        print('running in job array mode')
        parameters = pd.read_csv(args.parameters_file,sep='\t')
        createFolder(args.output_file)
        run_NMF_parameter_search(parameters, dataset, args.objective, max_iter=args.max_iter, report_freq=args.report_frequency, tol_=args.tolerance,
                                 active_thresh=1e-5,output_directory=args.output_file)


if __name__ == "__main__":
    main()
