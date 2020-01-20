#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
JSON_PYTORCH_MODULE_PATH="${DIR}/../pytorch-module-in-json"
TOOLBOX_PATH="${DIR}/../leelab-toolbox"
STRFLAB_PATH="${DIR}/../strflab-python"
# use https://files.pythonhosted.org/packages/82/5d/8bcd5b62399ed3de192bf03deaf6013fcad82c79338c362f7c2240887dd6/onnx2keras-0.0.18.tar.gz
# plus my fix in 3rdparty/pooling_layers.py, 3rdparty/activation_layers.py, 3rdparty/layers.py
ONNX2KERAS_PATH="${DIR}/../onnx2keras-0.0.18"
# I don't need other pythonpath (appending $PYTHONPATH)
# use commit 1eec6f1e60444f593edf9d7dd70f0ac4d4056181
# https://github.com/raghakot/keras-vis/archive/1eec6f1e60444f593edf9d7dd70f0ac4d4056181.tar.gz
KERASVIS_PATH="${DIR}/../keras-vis"
export PYTHONPATH="${DIR}:${JSON_PYTORCH_MODULE_PATH}:${TOOLBOX_PATH}:${STRFLAB_PATH}:${ONNX2KERAS_PATH}:${KERASVIS_PATH}"
