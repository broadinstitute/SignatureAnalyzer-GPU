# SignatureAnalyzer-GPU

# Installation
```
git clone https://github.com/broadinstitute/SignatureAnalyzer-GPU.git
```

# Setup
For easy set up you can create a python virtual enviroment which matches our own: 
```
$ python3 -m venv venv/

$ source venv/bin/activate .

(venv)$ pip install -r requirements-py3.txt
```

# How to run a single decomposition
SignatureAnalyzer runs on a count matrix (passed to the argument --data) and performs regularized NMF (Bayes NMF). You can specify the regularization you want on the resulting W and H matrices by using the arguments --prior_on_W and --prior_on_H . Passing "L1" is equivalent to an exponential prior and "L2" is half-normal.

For mathematical details see: https://arxiv.org/pdf/1111.6085.pdf. 

Note that as part of this work we derived the form for a mixed prior (e.g. L1 on W and L2 on H) see the supplemental note in the repo. 

Example command line for a single run of SignatureAnalyzer-GPU:
```
$ python SignatureAnalyzer-GPU.py --data data_matrix --max_iter=100000 --output_file output_file_stem --prior_on_W L1 --prior_on_H L2
```
Data should be formatted NxM where M is the number of channels and N is the number of samples


# Perform a parameter search or perform multiple restarts
