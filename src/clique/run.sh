BIN=../../bin/clique_GM_LUT
GRAPH=../../inputs/cit-Patents/graph
export CUDA_VISIBLE_DEVICES=7
${BIN} ${GRAPH} $1
