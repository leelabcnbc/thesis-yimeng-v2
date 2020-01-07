# should be upload as leelabcnbc/yimeng-thesis-v2:20200106
#  $ docker build . -t leelabcnbc/yimeng-thesis-v2:20200106
#  $ docker run --rm -it leelabcnbc/yimeng-thesis-v2:20200106
#  $ docker push leelabcnbc/yimeng-thesis-v2:20200106

# for nvidia-docker, you need run like the following
# sudo docker run -it --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all -e NVIDIA_VISIBLE_DEVICES=all leelabcnbc/yimeng-thesis-v2:20200106

FROM continuumio/miniconda3:4.7.12
# use this one, NOT continuumio/miniconda3:4.6.14-alpine,
# which has no bash
# <https://github.com/ContinuumIO/docker-images/issues/120>

RUN conda create -n leelab -c conda-forge -y python==3.7.6 \
    # four big ones.
    numpy==1.17.3 scipy==1.4.1 matplotlib==3.1.2 pandas==0.25.3 \
    # other ones in my "default" scripts
    notebook==6.0.1 h5py==2.10.0 git==2.24.0 \
    # additional pytorch deps and extra packages.
    scikit-image==0.16.2 scikit-learn==0.22 joblib==0.14.1 \
    # tf. 1.15 does not work, giving core dump for keras-vis and sometimes keras.
    tensorflow-gpu==1.14.0 \
    # psutil for <https://github.com/leelabcnbc/thesis-yimeng-v1/issues/20>
    psutil==5.6.7 \
    && conda clean --all

# pytorch
RUN /opt/conda/envs/leelab/bin/pip install --no-cache-dir \
    torch==1.3.1 \
    torchvision==0.4.2

# keras
RUN /opt/conda/envs/leelab/bin/pip install --no-cache-dir \
    keras==2.3.1

ENV PATH /opt/conda/envs/leelab/bin:$PATH
# for nice interative shell
RUN echo "conda activate leelab" >> ~/.bashrc
