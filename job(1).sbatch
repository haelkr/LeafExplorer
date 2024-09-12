#!/bin/bash
#SBATCH -A naiss2023-22-395
#SBATCH -n 1
#SBATCH -t 50:00:00
#SBATCH --gpus-per-task=1

module load Anaconda/2021.05-nsc1
conda create -n plant python=3.7
conda activate plant
conda install tensorflow=1.14
conda install tensorflow-gpu=1.14
conda install keras=2.2.4
conda install scikit-image==0.16.2
conda install h5py==2.10.0
conda install -c anaconda ipython
conda install opencv

#conda install cython

#here comes the main file


python C:\Users\Exjobb\Downloads\Mask_RCNN\myfile.py
#execute with sbatch filename to run on GPU not CPU

#for the test.py

conda install pandas
conda install seaborn
conda install scikit-learn

#for the preprocessing used analysis environment

#python 3.10
#skicit 0.19


#-------------------------commands:
#activate environments with 
#conda activate F:\Hannah\plant
#conda activate F:\Hannah\temperature

cd C:\Users\Exjobb\Downloads\Code_folder
conda activate F:\Hannah\temperature

python step1_preprocess.py
python step2_temperature.py
python step3_analysis.py


python temperature_on_coco.py
python training_on_coco.py
python evaluating_on_coco.py
python plant_analysis.py

tensorboard --logdir="F:\\Hannah\\logs"

#for the detection and temperature analysis use temperature, with packages for plantcv and maskrcnn

#temperature (analysis on mobaxterm)
# conda create -n analysis python=3.10
# conda install tensorflow==2.8
# conda install keras==2.8.0

#conda install scikit-image==0.19
#conda install h5py==3.8

#conda install -c conda-forge tqdm
# beforehand numpy 1.24.3, now numpy 1.23.1
# conda install numpy==1.23.1
#conda install pip
#conda install cython
#conda install IPython
#conda install -c conda-forge ipython
#conda install pandas
#conda install seaborn
# conda config --add channels conda-forge
# conda install plantcv


# not in analysis:

#conda install numba
#pip install tensorflow-gpu==2.8
#conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
#conda install spyder






#plantcv requirements:
# python 3.10
# matplotlib>=1.5,<3.6
# numpy>=1.11,<1.23
# pandas<2
# python-dateutil
# scipy
# scikit-image>=0.19
# scikit-learn
# plotnine
# dask
# dask-jobqueue
# opencv-python
# statsmodels

#right now I am running a tensorflow2.8 keras 2.8 implementation from github https://github.com/Densus100/Mask-RCNN-TF2.8-keras2.8
#change between mrcnntf1 and mrcnntf2