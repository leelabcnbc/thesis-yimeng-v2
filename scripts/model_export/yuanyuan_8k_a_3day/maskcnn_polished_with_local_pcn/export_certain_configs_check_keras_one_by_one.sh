#!/usr/bin/env bash

# this script run the checking of each model in an individual script,
# as running all model checking together somehow can make the program stuck.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


for i in {0..2303}
do
  echo "${i}"
  python "${DIR}/export_certain_configs_check_keras.py" "${i}"
done
