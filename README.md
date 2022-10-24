# How to run Yimeng's project

This documentation guides you through some of the basic setups for training neural prediction models on mind cluster using Yimeng's framework.
> You should also read the [original v2 doc](https://github.com/leelabcnbc/thesis-yimeng-v2/blob/master/README.md) to get a general idea of what you need to do. 

## Environment setup
### Dependencies
The main project is hosted  [here](https://github.com/leelabcnbc/thesis-yimeng-v2), you should clone it as well as its following dependencies into your workspace:
 - [`strflab-python`](https://github.com/leelabcnbc/strflab-python)
 - [`pytorch-module-in-json`](https://github.com/leelabcnbc/pytorch-module-in-json/tree/master)
 - [`gaya-data`](https://github.com/leelabcnbc/gaya-data/tree/master)
 > The dependencies should locate in the same directory as the main project
### Disk image
Copy `/home/sichengd/toolchain` to your home directory. This folder contains a disk image in which the models will be trained on compute nodes using Singularity.
>Singularity is already installed on the compute nodes of mind cluster, so you don't need to worry about that.
### Data
The data files are not hosted on GitHub since they are too large, so you need manually copy them to the corresponding directories in your workspace.
 - 8K:
 `yuanyuan_8k_neural.hdf5` and `yuanyuan_8k_images.hdf5` under `/user_data/yimengzh/thesis-yimeng-v2/results/datasets/raw/`
 - NS2250:
 `/user_data/yimengzh/gaya-data/data/tang/batch/final/tang_neural.npy` and `/user_data/yimengzh/gaya-data/data/tang/images/all_imags.npy`
 - CB19:
 `/user_data/sichengd/thesis-yimeng-v2/private_data/Cadena_PlosCB19_data`
 - Tang V1:
 `/user_data/sichengd/thesis-yimeng-v2/private_data/tang_v1`
### Python
 The project originally runs on PyTorch 1.3.1. You can create your Python environment however you like but here I provide a simple way to do that using conda:
  1. Download miniconda [here](https://docs.conda.io/en/latest/miniconda.html#linux-installers) and install it into your home directory
  2. Create a conda environment by running
 `conda env create -f /user_data/sichengd/thesis-yimeng-v2/env.yml`
  3. You can then activate the environment by
`conda activate yimeng`
## Training models
To train models under Yimeng's framework, normally you follow these steps:
1. Define a hyperparameter space that you want to search through
2. Generate training scripts, one for each combination of hyperparameters
3. Submit jobs to SLURM scheduler on mind cluster
4. After training, check whether the models are trained properly
### Define a hyperparameter space
The framework defines hyperparameter configurations as dictionaries and uses an iterator to generate all the combinations. The configurations are defined in `$ROOT/thesis_v2/configs/model/`  with a separate file for each different type of models. When you have in mind a specific hyperparameter space that you want to do experiments on, you should first check whether it's already defined in the corresponding file, in which case you can simply reuse it, or otherwise you need to create it.

The configurations used for generating results in the neurips paper are defined in `$ROOT/thesis_v2/configs/model/maskcnn_polished_with_rcnn_k_bl.py` as:
 - `main_models_8k_generator`
 - `main_models_cb19_generator`
 - `main_models_ns2250_generator`
> `$ROOT` refers to the root directory of the project in your workspace
### Generate training scripts and submit jobs to SLURM
Once you have your hyperparameter configuration ready, follow these steps to generate training scripts:
1. Create a script under `$ROOT/scripts/training/...` which imports the configuration and generate the training scripts accordingly. 
2. Run the following command under `$ROOT` to setup `PYTHONPATH`:
`source setup_env_variables.sh`
3. Run the script you just created. When it finishes running, a number of sbatch scripts should already be created under `$ROOT/trash/...`, now you can either press Ctrl^D to exit or press Enter to directly submit all the jobs to SLURM. If you choose to exit, you can use the `sbatch` command to submit the jobs later.
> SLURM is a job scheduler to manage computing resources on mind cluster. If you would like to use CPUs and/or GPUs to run your job, you have to request through SLURM. Check out the documentation [here](https://ni.cmu.edu/computing/knowledge-base/slurm-scheduler/).
### Check how the training goes
After submitting your jobs to SLURM, you can use the `squeue` command to get a squeue of jobs that are either running or pending for resources. If your job disappears from the queue, it either finished running successfully or it failed, so it's important to check regularly whether your models are trained properly as expected because you want to re-run the models that failed. To do this, you can either check the timestamps created in `$ROOT/results/models` where all the trianed models are stored, or you can also check the training log. For an example of a script to check whether the models are successfully trained, refer to `/user_data/sichengd/thesis-yimeng-v2/trash/check.py`.
