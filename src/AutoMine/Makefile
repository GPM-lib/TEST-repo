include ../common.mk
#INCLUDES+=-I$(CUB_DIR)
INCLUDES+=-I./cpu_kernels
all: cpu_serials 

cpu_serials: $(OBJS) cpu_serials.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) cpu_serials.o $(OBJS) -o $@ -lgomp
	#mv $@ $(BIN)

clean:
	rm *.o
