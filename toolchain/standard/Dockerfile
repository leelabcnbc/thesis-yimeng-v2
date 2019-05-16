# should be upload as leelabcnbc/yimeng-thesis-v2:standard (no date)
#  $ docker build . -t leelabcnbc/yimeng-thesis-v2:standard
#  $ docker run --rm -it leelabcnbc/yimeng-thesis-v2:standard
#  $ docker push leelabcnbc/yimeng-thesis-v2:standard

# for nvidia-docker, you need run like the following
# sudo docker run -it --rm --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=all -e NVIDIA_VISIBLE_DEVICES=all leelabcnbc/yimeng-thesis-v2:standard

FROM continuumio/miniconda3:4.6.14
# use this one, NOT continuumio/miniconda3:4.6.14-alpine,
# which has no bash
# <https://github.com/ContinuumIO/docker-images/issues/120>

RUN conda create -n leelab -c conda-forge -y python==3.7.3 \
    # four big ones.
    numpy==1.16.3 scipy==1.2.1 matplotlib==3.0.3 pandas==0.24.2 \
    # other ones in my "default" scripts
    notebook==5.7.8 h5py==2.9.0 git==2.21.0 \
    # additional pytorch deps and extra packages.
    scikit-image==0.15.0 scikit-learn==0.21.1 joblib==0.13.2 \
    # sporco related deps
    imageio==2.5.0 numexpr==2.6.9 \
    pyfftw==0.11.1 future==0.17.1 \
    # cuda10 libs for cupy
    cudatoolkit==10.0.130 \
    # psutil for <https://github.com/leelabcnbc/thesis-yimeng-v1/issues/20>
    psutil==5.6.2 \
    && conda clean --all

## cuda10
# seems not working
# https://github.com/conda-forge/cudatoolkit-dev-feedstock/issues/7
#RUN conda install -n leelab -c conda-forge --no-update-deps cudatoolkit-dev==10.0 \
#    && conda clean --all

# sporco
RUN /opt/conda/envs/leelab/bin/pip install --no-cache-dir sporco==0.1.11 \
    cupy-cuda100==6.0.0

# pytorch
RUN /opt/conda/envs/leelab/bin/pip install --no-cache-dir \
    https://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-linux_x86_64.whl \
    torchvision==0.2.2.post3

# for centos6 (CNBC psych-o)
RUN mkdir /my_data /my_data_1 /my_data_2 /my_data_3 /my_data_4

ENV PATH /opt/conda/envs/leelab/bin:$PATH
# for nice interative shell
RUN echo "conda activate leelab" >> ~/.bashrc
