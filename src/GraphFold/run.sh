BIN=../../bin/pattern_gpu_GF_LUT
GRAPH=../../inputs/soc-delicious/graph
export CUDA_VISIBLE_DEVICES=7
${BIN} ${GRAPH} $1 $2
