BIN=../../bin/pattern_gpu_GM
GRAPH=../../inputs/cit-Patents/graph
export CUDA_VISIBLE_DEVICES=7
${BIN} ${GRAPH} $1