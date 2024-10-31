#!/bin/bash

# check CUDA info
#echo "${CUDA_PATH}"
#echo "${CUDA_HOME}"

export PYTHONPATH=${PYTHONPATH}:${PROJECT_PATH}

TASK_PATH=${PROJECT_PATH}/task/${TASK_NAME}

echo "${TASK_PATH}"

args="--repo_path ${PROJECT_PATH} --task_name ${TASK_NAME} --task_type ${TASK_TYPE} --model_name ${MODEL_NAME} --config_name ${CONFIG_NAME}"

echo "args: ${args}"

if [ "${TASK_TYPE}" = "data_preparation" ]; then
  python "${TASK_PATH}"/main_data_preparation.py ${args}
elif [ "${TASK_TYPE}" = "model_train" ]; then
  python "${TASK_PATH}"/main_model_train.py ${args}
elif [ "${TASK_TYPE}" = "model_evaluation" ]; then
  python "${TASK_PATH}"/main_model_evaluation.py ${args}
else
  echo "Invalid TASK_TYPE: ${TASK_TYPE}"
  exit 1
fi
