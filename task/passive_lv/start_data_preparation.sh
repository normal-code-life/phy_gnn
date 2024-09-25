#!/bin/bash

export TASK_NAME="passive_lv"
export MODEL_NAME="fe_heart_sage_v2"

export PROJECT_PATH="$(cd `dirname $0`/../../; pwd)"
echo "project root path: ${PROJECT_PATH}"

export CONFIG_NAME="train_config"
export TASK_TYPE="data_preparation"

## setup NUMBA info
#export NUMBA_NUM_THREADS=16
#echo "${NUMBA_NUM_THREADS}"

sh "${PROJECT_PATH}/common/sbin/main_process.sh"
