#!/usr/bin/env bash

# https://cis.gvsu.edu/~meos/shell/scripts.html
set -o errexit
set -o nounset



# https://stackoverflow.com/a/246128/3692822
# https://stackoverflow.com/questions/59895/getting-the-source-directory-of-a-bash-script-from-within
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PWD_PREV="$(pwd)"
# get CRCNS data from sparrowhawk
rsync -avP yimengzh@sparrowhawk.cnbc.cmu.edu:/datasmart/leelab/standard_datasets/crcns_pvc-8 "${DIR}/private_data/"
# then checksum
cd "${DIR}/private_data/crcns_pvc-8"
md5sum -c checksums.md5
cd "${PWD_PREV}"

#if [[ "$(hostname)" == "psych-o.hpc1.cs.cmu.edu" ]]; then
#    # 8k
#    ln -s /data2/leelab/cnn-model-leelab-8000/data/processed/leelab_8000 "${DIR}/private_data/yuanyuan_8k"
#    ln -s /data2/leelab/yuanyuan/data/leelab_8000 "${DIR}/private_data/yuanyuan_8k_raw"
#    ln -s /data2/leelab/cnn-model-leelab-8000/data_processing/leelab_8000 "${DIR}/private_data/yuanyuan_8k_preprocessing"
#else
    # rsync 8k stuffs.

# since I use container, softlink won't work. so just copy.
mkdir -p "${DIR}/private_data/yuanyuan_8k"
rsync -avP yimengzh@psych-o.hpc1.cs.cmu.edu:/data2/leelab/cnn-model-leelab-8000/data/processed/leelab_8000/ "${DIR}/private_data/yuanyuan_8k/"

mkdir -p "${DIR}/private_data/yuanyuan_8k_raw"
rsync -avP yimengzh@psych-o.hpc1.cs.cmu.edu:/data2/leelab/yuanyuan/data/leelab_8000/ "${DIR}/private_data/yuanyuan_8k_raw/"

mkdir -p "${DIR}/private_data/yuanyuan_8k_preprocessing"
rsync -avP yimengzh@psych-o.hpc1.cs.cmu.edu:/data2/leelab/cnn-model-leelab-8000/data_processing/leelab_8000/ "${DIR}/private_data/yuanyuan_8k_preprocessing/"
#fi

# this is for decompressing thoese imagenet raw images for yuanyuan 8k
mkdir -p "${DIR}/private_data/yuanyuan_8k_raw_images"
tar -xvf "${DIR}/private_data/yuanyuan_8k_raw/ImageNet_25k/rp_252.tar" -C "${DIR}/private_data/yuanyuan_8k_raw_images"

echo "DONE!"
