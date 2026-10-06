# hyper: inference engine for 3x RTX 3090 + Threadripper (sm_86, AVX2)
NVCC     ?= /usr/local/cuda-12.9/bin/nvcc
CXX      ?= g++
LLAMA    ?= $(HOME)/llama.cpp-src
ARCH     := -gencode arch=compute_86,code=sm_86
CXXFLAGS := -O3 -march=native -std=c++17 -Wall -Wno-unused-function -fopenmp
NVFLAGS  := -O3 -std=c++17 $(ARCH) -lineinfo --use_fast_math -Xcompiler "-O3 -march=native -fopenmp"
LIBS     := -L/usr/local/cuda-12.9/lib64 -lcudart -lcublas -lcuda -lgomp -lpthread -ldl

BUILD := build
SRC   := $(wildcard src/*.cpp)
CU    := $(wildcard src/*.cu)
OBJ   := $(SRC:src/%.cpp=$(BUILD)/%.o) $(CU:src/%.cu=$(BUILD)/%.o)

all: $(BUILD)/hyper $(BUILD)/ref $(BUILD)/arbench

$(BUILD)/%.o: src/%.cpp src/*.h
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/%.o: src/%.cu src/*.h src/*.cuh
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) -c $< -o $@

$(BUILD)/hyper: $(OBJ) tools/hyper_main.cpp
	$(NVCC) $(NVFLAGS) -Isrc tools/hyper_main.cpp $(OBJ) -o $@ $(LIBS)

# golden reference against mainline llama.cpp (static libs from $(LLAMA)/build)
LLAMA_LIBS := $(LLAMA)/build/src/libllama.a $(LLAMA)/build/ggml/src/ggml-cuda/libggml-cuda.a \
              $(LLAMA)/build/ggml/src/libggml-cpu.a $(LLAMA)/build/ggml/src/libggml.a $(LLAMA)/build/ggml/src/libggml-base.a
$(BUILD)/ref: tools/ref.cpp
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -I$(LLAMA)/include -I$(LLAMA)/ggml/include tools/ref.cpp -o $@ \
	  -Wl,--start-group $(LLAMA_LIBS) -Wl,--end-group $(LIBS)

clean:
	rm -rf $(BUILD)

.PHONY: all clean

$(BUILD)/arbench: tools/arbench.cu $(BUILD)/kernels.o
	$(NVCC) $(NVFLAGS) tools/arbench.cu $(BUILD)/kernels.o -o $@ $(LIBS)

$(BUILD)/gemvbench: tools/gemvbench.cu $(BUILD)/kernels.o
	$(NVCC) $(NVFLAGS) tools/gemvbench.cu $(BUILD)/kernels.o -o $@ $(LIBS)

$(BUILD)/mmabench: tools/mmabench.cu $(BUILD)/kernels.o
	$(NVCC) $(NVFLAGS) tools/mmabench.cu $(BUILD)/kernels.o -o $@ $(LIBS)

$(BUILD)/arbulk: tools/arbulk.cu $(BUILD)/kernels.o
	$(NVCC) $(NVFLAGS) tools/arbulk.cu $(BUILD)/kernels.o -o $@ $(LIBS)

# OpenAI-compatible server: chat templates / tool-call parsing from mainline libcommon, vocab from libllama
COMMON_LIBS := $(LLAMA)/build/common/libllama-common.a $(LLAMA)/build/common/libllama-common-base.a \
               $(LLAMA)/build/vendor/cpp-httplib/libcpp-httplib.a
$(BUILD)/hyper-server: $(OBJ) tools/hyper_server.cpp
	$(NVCC) $(NVFLAGS) -Isrc -I$(LLAMA)/include -I$(LLAMA)/ggml/include -I$(LLAMA)/common -I$(LLAMA)/vendor \
	  tools/hyper_server.cpp $(OBJ) -o $@ \
	  -Xlinker --start-group $(COMMON_LIBS) $(LLAMA_LIBS) -Xlinker --end-group $(LIBS) -lssl -lcrypto
