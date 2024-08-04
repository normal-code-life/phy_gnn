#!/bin/bash

export TASK_NAME="passive_biv"

export PROJECT_PATH="$(cd `dirname $0`/../../; pwd)"
echo "project root path: ${PROJECT_PATH}"

export CONFIG_NAME="train_config"
export TASK_TYPE="model_train"

sh "${PROJECT_PATH}/common/sbin/main_process.sh"
