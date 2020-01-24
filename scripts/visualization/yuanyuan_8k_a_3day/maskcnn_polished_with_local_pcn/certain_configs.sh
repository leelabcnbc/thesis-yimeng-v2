#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

python "${DIR}/certain_configs.py" 0.2 0.1 0
python "${DIR}/certain_configs.py" 1.0 0.1 0
python "${DIR}/certain_configs.py" 0.02 0.1 0

python "${DIR}/certain_configs.py" 0.02 0.5 0
python "${DIR}/certain_configs.py" 0.02 0.02 0
python "${DIR}/certain_configs.py" 0.02 1.0 0


python "${DIR}/certain_configs.py" 0.2 0.5 0
python "${DIR}/certain_configs.py" 0.2 0.02 0
python "${DIR}/certain_configs.py" 0.2 1.0 0
