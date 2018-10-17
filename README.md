# SignatureAnalyzer-GPU

# Setup
For easy set up you can create a python virtual enviroment which matches our own: 
```
$ python3 -m venv venv/

$ source venv/bin/activate .

(venv)$ pip install -r requirements-py3.txt
```
# How to run

```
$ python run_ARD_NMF.py --data data_matrix --max_iter=100000 --a=9 --output_file output_file_stem --prior_on_W L1|L2 --prior_on_H L1|L2
```
Data should be formatted NxM where M is the number of channels and N is the number of samples
