#!/bin/bash

# check CUDA info
echo "${CUDA_PATH}"
echo "${CUDA_HOME}"

export PYTHONPATH=${PYTHONPATH}:${PROJECT_PATH}

TASK_PATH=${PROJECT_PATH}/task/${TASK_NAME}

echo "${TASK_PATH}"

python ${TASK_PATH}/main.py --repo_path "${PROJECT_PATH}" --task_name "${TASK_NAME}" --config_name "${CONFIG_NAME}"


