BIN=../../bin/pattern_gpu_GM
GRAPH=../../inputs/mico/graph
export CUDA_VISIBLE_DEVICES=0
${BIN} ${GRAPH} $1
