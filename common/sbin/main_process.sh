#!/bin/bash

# check CUDA info
echo "${CUDA_PATH}"
echo "${CUDA_HOME}"

export PYTHONPATH=${PYTHONPATH}:${PROJECT_PATH}

TASK_PATH=${PROJECT_PATH}/task/${TASK_NAME}

echo "${TASK_PATH}"

if [ "${TASK_TYPE}" = "data_preparation" ]; then
  python "${TASK_PATH}"/main_data_preparation.py --repo_path "${PROJECT_PATH}" --task_name "${TASK_NAME}" --config_name "${CONFIG_NAME}"
elif [ "${TASK_TYPE}" = "model_train" ]; then
  python "${TASK_PATH}"/main_model_train.py --repo_path "${PROJECT_PATH}" --task_name "${TASK_NAME}" --config_name "${CONFIG_NAME}"
else
  echo "Invalid TASK_TYPE: ${TASK_TYPE}"
  exit 1
fi
