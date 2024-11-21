BIN=../../bin/pattern_gpu_GM_LUT
GRAPH=../../inputs/mico/graph
export CUDA_VISIBLE_DEVICES=0
${BIN} ${GRAPH} $1
