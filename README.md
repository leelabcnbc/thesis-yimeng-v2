# thesis-yimeng-v2

This file is a revision of the [original doc](./README.old.md).

The current version only contains doc related to reproducing results
in the [paper draft](https://www.overleaf.com/read/cpkywzzdhrsj).

`$ROOT` refers to repo root.

## dependencies

* `yimeng-thesis-v2-20200106_2c0c603d8a871cd40d99848371ad443a.simg` under `~/toolchain/`. it can be obtained
  by converting the docker image available at
  [`docker pull leelabcnbc/yimeng-thesis-v2:20200106`](https://hub.docker.com/layers/leelabcnbc/yimeng-thesis-v2/20200106/images/sha256-6aa6babb9241a06839f0da8c4290c13677354eae796adfc2edb4177d5d7d9e15?context=repo).
  Check Section *toolchain 20200106* in the old README.
* [Singularity](https://github.com/hpcng/singularity). should work on 2.6.1 as well as 3.0 version.
* you need those some of the dependencies specified in `$ROOT/setup_env_variables.sh`. Those packages
are mostly available in lab GitHub. Only the following of them are needed to reproduce the results in the paper.
Click each link below for each dependency's commit that worked with this repo. Newer commits in theory should do as well.
    * [`pytorch-module-in-json`](https://github.com/leelabcnbc/pytorch-module-in-json/tree/083cef7d2dc688b0889b16c5085625d421dfa8a1)
      which implements the DSL for model definition.
    * [`strflab-python`](https://github.com/leelabcnbc/strflab-python/tree/34d6fbe1e79f07a9469ab86fb6a57a6a99fded79)
      for computing ccnorm.
    * [`gaya-data`](https://github.com/leelabcnbc/gaya-data/tree/761ae2aa88e37d8da9eeb5a09ad0b249a324a0c0) needed to obtain NS 2250 data.
      Check with [Hal](https://github.com/hal-rock) on the location of the data.

## reproduce results

The steps should work on the CNBC cluster (mind) and will work with single machine
with some small adaptations.

All the actual computation is done inside the Singularity container.

* For model training, explicit invocation of Singularity is not needed, as my code
already handles that.
* For everything else, the code has to run after doing the following steps.
    1. open the container.
       ```
       singularity shell --nv -B /data2/yimengzh:/my_data -B /scratch:/my_data_2 ~/toolchain/yimeng-thesis-v2-20200106_2c0c603d8a871cd40d99848371ad443a.simg
       ```
    2. set up environment variables
       ```
       cd /my_data
       # note the starting `.` you can also do `source ./setup_env_variables.sh`
       . ./setup_env_variables.sh
       ```
    3. this is only needed for Jupyter notebooks.
       ```
       # XXXX should be replaced by an appropriate port number.
       jupyter notebook --no-browser --port=XXXX
       ```

### preprocess neural data

#### ImageNet 8K

1. first, you need to download ImageNet 8K data. Run the command OUTSIDE the container.
   ```
   $ROOT/setup_private_data.sh
   ```
2. run the following inside the container
   ```
   python $ROOT/scripts/preprocessing/raw_data.py
   python $ROOT/scripts/preprocessing/prepared_data.py
   ```

#### NS 2250

Ask Hal about it. This code repo uses Hal's code under the hood
to obtain the data.

### model training

All commands should run outside the container, with a basic Python 3.6+ environment
without any additional dependency needed. On the CNBC cluster, such an environment
can be established using `scl enable rh-python36 bash`.

#### main models (recurent and feed-forward, no ablation)

##### ImageNet 8K

Run the following files under `$ROOT/scripts/training/yuanyuan_8k_a_3day/maskcnn_polished_with_rcnn_k_bl`.
These files in total will train some extra models. But these form the minimal set
of files required to cover all models used in the paper.

* `submit_20200530.py`
* `submit_20200530_2.py`
* `submit_20200617.py`
* `submit_20200704.py`
* `submit_20200705.py`
* `submit_20200707.py`
* `submit_20200708.py`
* `submit_20200709.py`
* `submit_20200731.py`
* `submit_20201001.py`
* `submit_20201012.py`

##### NS 2250


Run the following files under `$ROOT/scripts/training/gaya/maskcnn_polished_with_rcnn_k_bl`.
These files in total will train some extra models. But these form the minimal set
of files required to cover all models used in the paper.

* `submit_20201002_tang.py`
* `submit_20201018_tang.py`

#### multi path models that correspond to recurrent models

Only 8/16/32 ch models were considered; higher ch will result
in a higher frequency of OOM, making the results not very useful.

##### ImageNet 8K

TBD

##### NS 2250

TBD
