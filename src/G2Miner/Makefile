include ../common.mk
#INCLUDES+=-I$(CUB_DIR)
INCLUDES+=-I./gpu_GM_LUT_kernels
all: pattern_gpu_GM_LUT

pattern_gpu_GM_LUT: $(OBJS) gpu_GM_LUT.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) gpu_GM_LUT.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
