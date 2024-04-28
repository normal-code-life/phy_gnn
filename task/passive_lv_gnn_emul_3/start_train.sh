#!/bin/bash

export TASK_NAME="passive_lv_gnn_emul_3"

export PROJECT_PATH="$(cd `dirname $0`/../../; pwd)"
echo "project root path: ${PROJECT_PATH}"

sh "${PROJECT_PATH}/common/sbin/model_train_pipeline.sh"
