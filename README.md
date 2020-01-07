# thesis-yimeng-v2
good parts of thesis-yimeng-v1, better refactoring.

`$ROOT` refers to repo root.

## dependencies

you also need those dependencies specified in `$ROOT/setup_env_variables.sh`. Those packages
are mostly available in lab GitHub.

## set up toolchain (only once)

On May 16, 2019 (Eastern Time), I ran this on my MacBook Pro (steps 1-3), and
`yimengzh2080ti.cnbc.cmu.edu` (steps 3-6).

1. under `$ROOT/toolchain/standard`, run
    ~~~
    docker build . -t leelabcnbc/yimeng-thesis-v2:standard
    ~~~
2. then run `docker save --output yimeng-thesis-v2.tar leelabcnbc/yimeng-thesis-v2:standard` to save the image to a tar file.
   last time I ran it, MD5 was `20640abe170a90a7d323165ed8de0c5b`
3. then upload that file to somewhere with singularity
   (tested on 2.6.1, which is compatible with 2.5.0 on CNBC psych-o cluster)
4. run `sudo docker load -i yimeng-thesis-v2.tar` to get it.
5. check <https://github.com/sylabs/singularity/issues/1537#issuecomment-388642244>
   and <https://github.com/sylabs/singularity/issues/1537#issuecomment-402823767>
    ~~~
    # Start a docker registry
    docker run -d -p 5000:5000 --restart=always --name registry registry:2
    # Push local docker container to it
    docker tag leelabcnbc/yimeng-thesis-v2:standard localhost:5000/yimeng-thesis-v2:standard
    docker push localhost:5000/yimeng-thesis-v2:standard
    # build to get `yimeng-thesis-v2.simg`
    sudo SINGULARITY_NOHTTPS=1 /opt/singularity-2.6.1/bin/singularity build yimeng-thesis-v2.simg docker://localhost:5000/yimeng-thesis-v2:standard
    ~~~
   last time I ran this, I get md5 of `34da8d5eac5297d9fafa5e8be3c635f0`.
6. rename the file according to the MD5. For me, it's
   `yimeng-thesis-v2_34da8d5eac5297d9fafa5e8be3c635f0.simg`.

## run toolchain

### `yimengzh2080ti.cnbc.cmu.edu`

```
/opt/singularity-2.6.1/bin/singularity shell --nv -B /data2:/my_data -B /data1:/my_data_2 -B /run:/run ~/toolchain/yimeng-thesis-v2_34da8d5eac5297d9fafa5e8be3c635f0.simg
```

### CNBC cluster

```
singularity shell --nv -B /data2/yimengzh:/my_data -B /scratch:/my_data_2 ~/toolchain/yimeng-thesis-v2_34da8d5eac5297d9fafa5e8be3c635f0.simg
```

## toolchain 20200106

This is a new toolchain image with PyTorch 1.3.1 and TensorFlow 1.14.0. Check  `$ROOT/toolchain/20200106`.
At late night January 6, 2020 (Pacific Time), I generated both the Docker image and the Singularity image on `yimengzh2080ti.cnbc.cmu.edu`

* The Docker image tar has MD5 `b72ebbd5efb5bc2983d460ffca56ee02`
* The Singularity image name `yimeng-thesis-v2-20200106_726ed91c60c6f4ac02c87379f32d80e0.simg`

## download some data set

run `./setup_private_data.sh` OUTSIDE the container.


## reproduce results

unless otherwise mentioned, scripts should be run **IN** the container
(check **run toolchain** above), and
after `. $ROOT/setup_env_variables.sh` (notice the leading dot, which
is equivalent to `source` in `bash`) has been run.

### preprocess image and neural data

#### preprocess raw data

`python $ROOT/scripts/preprocessing/raw_data.py`

#### preprocess prepared data

`python $ROOT/scripts/preprocessing/prepared_data.py`

### CNN feature extraction

#### 8k (a) data set, VGG networks

`python $ROOT/scripts/feature_extraction/yuanyuan_8k_a/vgg.py`

### CNN training

these code has been mostly has been designed to work on CNBC cluster, via Slurm job scheduler.
To work on a standard machine, you need do the following steps.

1. `ctrl+c` when the program prompts `press enter to optionally sbatch all scripts`.
2. run all shell scripts indicated by the (partially finished) program one by one.
    * use some script like <https://github.com/leelabcnbc/thesis-yimeng-v1/blob/master/gen_single_machine_batch_script.py>
      to generate some wrapper scripts, one for each GPU on the machine.


Unless otherwise indicated, the training scripts (those `submit.py` files) should be run **OUTSIDE** the container, in an
environment with 1) `h5py` and 2) `python 3.6+`. However, you still need to run `. $ROOT/setup_env_variables.sh` before
running these `submit.py` files. The reason for running OUTSIDE is because these training scripts interact
with Slurm on the host.


#### CNN 8k (a) dataset transfer learning, VGG networks.



`python $ROOT/scripts/training/yuanyuan_8k_a_3day/transfer_learning_factorized_vgg/submit.py`

##### analysis

check <https://github.com/leelabcnbc/thesis-yimeng-v2/blob/master/results_processed/yuanyuan_8k_a_3day/transfer_learning_factorized_vgg/vgg.ipynb>;
basically, the results matched those in <https://github.com/leelabcnbc/cnn-model-leelab-8000/blob/master/slides/20190107_Wei_ConvolutionalNeuralNetwork_2019a.pdf> (model 3, 3 day data);
the key to make performance match is using 2x downsampled images rather than 4x. Check some notes [here](https://github.com/leelabcnbc/thesis-yimeng-v1/issues/28#issuecomment-500072420) as well.
