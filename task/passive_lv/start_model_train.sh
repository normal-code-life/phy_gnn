#!/bin/bash

export TASK_NAME="passive_lv"
export MODEL_NAME="fe_heart_sage_v3"
export CONFIG_NAME="train_config"

if [ "${1}" = "-model_name" ]; then
  export MODEL_NAME="${2}"
  echo "model_name=${MODEL_NAME}"
else
  echo "no model_name pass in, will use default value ${MODEL_NAME}"
fi

if [ "${3}" = "-config_name" ]; then
  export CONFIG_NAME="${4}"
  echo "config_name=${MODEL_NAME}"
else
  echo "no config_name pass in, will use default value ${CONFIG_NAME}"
fi

export PROJECT_PATH="$(cd `dirname $0`/../../; pwd)"
echo "project root path: ${PROJECT_PATH}"

export TASK_TYPE="model_train"


sh "${PROJECT_PATH}/common/sbin/main_process.sh"
