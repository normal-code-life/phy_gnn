#!/bin/bash

export TASK_NAME="passive_lv"
export MODEL_NAME="fe_heart_sage_v2"

if [ "${1}" = "--model_name" ]; then
  export MODEL_NAME="${2}"
  echo "model_name=${MODEL_NAME}"
else
  echo "no model_name pass in, will use default value ${MODEL_NAME}"
fi

export PROJECT_PATH="$(cd `dirname $0`/../../; pwd)"
echo "project root path: ${PROJECT_PATH}"

export CONFIG_NAME="train_config"
export TASK_TYPE="data_preparation"

## setup NUMBA info
#export NUMBA_NUM_THREADS=16
#echo "${NUMBA_NUM_THREADS}"

sh "${PROJECT_PATH}/common/sbin/main_process.sh"
