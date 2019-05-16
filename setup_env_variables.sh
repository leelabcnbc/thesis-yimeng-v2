#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
JSON_PYTORCH_MODULE_PATH="${DIR}/../pytorch-module-in-json"
TOOLBOX_PATH="${DIR}/../leelab-toolbox"
# I don't need other pythonpath (appending $PYTHONPATH)
export PYTHONPATH="${DIR}:${JSON_PYTORCH_MODULE_PATH}:${TOOLBOX_PATH}"
