#!/bin/bash

export PYTHONPATH=${PYTHONPATH}:${PROJECT_PATH}

TASK_PATH=${PROJECT_PATH}/task/${TASK_NAME}

python ${TASK_PATH}/main.py --repo_path "${PROJECT_PATH}" --task_name "${TASK_NAME}"


