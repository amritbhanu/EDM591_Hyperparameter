## EDM591_Hyperparameter
- Final PPT can be find here: https://tiny.cc/edm17ppt
- Final Report can be find here: https://tiny.cc/edm17report
- Github Repository can be found here: https://github.com/amritbhanu/EDM591_Hyperparameter

## Installation:
- Run pip install on requirements.txt to get the required packages.
- Need python2.7 

## Directory Structure:
- Data folder contains raw data as well as preprocessed data. You will need raw folder and datset2 folder inside. dataset1, dataset2, dataset3 csvs will automatically be generated.
- Dump folder contains the results dump of running our scripts. So that results can be generated quickly.
- Results folder contain all our graphs and results for the report and ppt. They will be automatically be generated our running our scripts.
- Src directory contains all our scripts.

## Src scripts:
- DE.py is the generalised code/class of our DE. We are planning to publish this DE as a python package.
- demos.py is the code which runs our main script by calling its function and parameters as argument.
- main.py is the main code which runs our tuned results and generates dump.
- ML.py is the generalised code of all our Machine learning implementations.
- Preprocess_dataset1_3.py is the preprocess script for dataset1 and dataset3.
- preprocessing_dataset2.py is the preprocess script for dataset2.
- read_pickle.py reads all the results dump from dump folder and generates graph in results folder.
- sk.py is the code for our statistical test which is scottknot.
- untuned.py is another main code which runs our untuned results and generates dump.

## How to run scripts:
Go into src folder and run in the sequential order, how we mentioned below.
1) 'python preprocessing_dataset2.py'
2) 'python Preprocess_dataset1_3.py'
3) 'python untuned.py _test dataset1' : this will generate dataset1_untuned.pickle in dump folder
4) 'python untuned.py _test dataset2' : this will generate dataset2_untuned.pickle in dump folder
5) 'python untuned.py _test dataset3' : this will generate dataset3_untuned.pickle in dump folder
6) Now to run these scripts you will need High Performance computing (HPC) servers since it will 4-8 hours to end each script. If it cant be run, we have provided the dump of our results. Directly jump to step 7.
    - 'python main.py _test dataset1' : this will generate dataset1.pickle and dataset1_late.pickle in dump folder
    - 'python main.py _test dataset2' : this will generate dataset2.pickle and dataset2_late.pickle in dump folder
    - 'python main.py _test dataset3' : this will generate dataset3.pickle and dataset3_late.pickle in dump folder
7) 'python read_pickle.py' : will generate graphs in results folder.


