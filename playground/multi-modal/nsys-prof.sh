#!/bin/bash

# 고정된 파라미터
# FIXED_PARAMS="-t nvtx,cuda,osrt,cudnn,cublas --capture-range-end stop -c cudaProfilerApi --cudabacktrace=true --stats=false -f true --backtrace dwarf"
FIXED_PARAMS="-t nvtx,cuda,cudnn,cublas --capture-range-end stop -c cudaProfilerApi --cudabacktrace=true --stats=true -f true --backtrace dwarf"

# --gpu-metrics-device {0 | 1 | 2 | 3 | all}
# -o {output_file_name}

# run.py examples
# ./nsys-prof.sh -o profile-run-py --gpu-metrics-device all python run.py --model 7b --dataset sharegpt --batch-size 2 --tensor-parallel-size 2 --seed 0

# 사용자 입력 파라미터와 함께 nsys profile 명령어 실행
nsys profile $FIXED_PARAMS --gpu-metrics-device 2 "$@"
