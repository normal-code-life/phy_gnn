#!/bin/bash

export TASK_NAME="passive_lv_gnn_emul"

export PROJECT_PATH="$(cd `dirname $0`/../../; pwd)"
echo "project root path: ${PROJECT_PATH}"

export CONFIG_NAME="train_config_lv_data"

sh "${PROJECT_PATH}/common/sbin/model_train_pipeline.sh"
