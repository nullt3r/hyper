# hyper: inference engine for 3x RTX 3090 + Threadripper (sm_86, AVX2)
NVCC     ?= /usr/local/cuda-12.9/bin/nvcc
CXX      ?= g++
LLAMA    ?= $(HOME)/llama.cpp-src
ARCH     := -gencode arch=compute_86,code=sm_86
CXXFLAGS := -O3 -march=native -std=c++17 -Wall -Wno-unused-function -fopenmp
NVFLAGS  := -O3 -std=c++17 $(ARCH) -lineinfo --use_fast_math -Xcompiler "-O3 -march=native -fopenmp" $(EXTRA)
LIBS     := -L/usr/local/cuda-12.9/lib64 -lcudart -lcublas -lcuda -lgomp -lpthread -ldl

BUILD := build
# v1 (qwen35 dense) objects; qwen4exp objects additionally need ggml (CPU experts)
OBJ   := $(BUILD)/gguf.o $(BUILD)/model.o $(BUILD)/kernels.o $(BUILD)/engine.o
OBJ4  := $(BUILD)/gguf.o $(BUILD)/model4.o $(BUILD)/kernels.o $(BUILD)/kernels4.o $(BUILD)/cpu_moe.o $(BUILD)/engine4.o \
         $(BUILD)/model5.o $(BUILD)/kernels5.o $(BUILD)/engine5.o
GGML_INC := -I$(LLAMA)/ggml/include -I$(LLAMA)/ggml/src

all: $(BUILD)/hyper $(BUILD)/ref $(BUILD)/arbench

$(BUILD)/%.o: src/%.cpp src/*.h
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) $(GGML_INC) -x cu -c $< -o $@

$(BUILD)/%.o: src/%.cu src/*.h src/*.cuh
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) $(GGML_INC) -c $< -o $@

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

# MoE routing kernel
$(BUILD)/routebench: $(OBJ4) tools/routebench.cu
	$(NVCC) $(NVFLAGS) -Isrc $(GGML_INC) tools/routebench.cu $(OBJ4) -o $@ \
	  -Xlinker --start-group $(LLAMA_LIBS) -Xlinker --end-group $(LIBS)

# MoE decode kernels (Flash-Next shapes)
$(BUILD)/moebench: $(OBJ4) tools/moebench.cu
	$(NVCC) $(NVFLAGS) -Isrc $(GGML_INC) tools/moebench.cu $(OBJ4) -o $@ \
	  -Xlinker --start-group $(LLAMA_LIBS) -Xlinker --end-group $(LIBS)

$(BUILD)/cpumoebench: $(OBJ4) tools/cpumoebench.cu
	$(NVCC) $(NVFLAGS) -Isrc $(GGML_INC) tools/cpumoebench.cu $(OBJ4) -o $@ \
	  -Xlinker --start-group $(LLAMA_LIBS) -Xlinker --end-group $(LIBS)

# dense GEMV on GGUF blocks: bandwidth on the Flash-Next K-quant shapes
$(BUILD)/ggbench: $(OBJ4) tools/ggbench.cu
	$(NVCC) $(NVFLAGS) -Isrc $(GGML_INC) tools/ggbench.cu $(OBJ4) -o $@ \
	  -Xlinker --start-group $(LLAMA_LIBS) -Xlinker --end-group $(LIBS)

# expert dequantization (deq8) vs ggml's reference
$(BUILD)/deqtest: $(OBJ4) tools/deqtest.cu
	$(NVCC) $(NVFLAGS) -Isrc $(GGML_INC) tools/deqtest.cu $(OBJ4) -o $@ \
	  -Xlinker --start-group $(LLAMA_LIBS) -Xlinker --end-group $(LIBS)

$(BUILD)/gemvbench: tools/gemvbench.cu $(BUILD)/kernels.o
	$(NVCC) $(NVFLAGS) tools/gemvbench.cu $(BUILD)/kernels.o -o $@ $(LIBS)

$(BUILD)/mkbench: tools/mkbench.cu $(BUILD)/kernels.o
	$(NVCC) $(NVFLAGS) tools/mkbench.cu $(BUILD)/kernels.o -o $@ $(LIBS)

$(BUILD)/mmabench: tools/mmabench.cu $(BUILD)/kernels.o
	$(NVCC) $(NVFLAGS) tools/mmabench.cu $(BUILD)/kernels.o -o $@ $(LIBS)

$(BUILD)/arbulk: tools/arbulk.cu $(BUILD)/kernels.o
	$(NVCC) $(NVFLAGS) tools/arbulk.cu $(BUILD)/kernels.o -o $@ $(LIBS)

# OpenAI-compatible server: chat templates / tool-call parsing from mainline libcommon, vocab from libllama
COMMON_LIBS := $(LLAMA)/build/common/libllama-common.a $(LLAMA)/build/common/libllama-common-base.a \
               $(LLAMA)/build/vendor/cpp-httplib/libcpp-httplib.a
OBJ_ALL := $(sort $(OBJ) $(OBJ4))
$(BUILD)/hyper-server: $(OBJ_ALL) tools/hyper_server.cpp tools/chat_page.h
	$(NVCC) $(NVFLAGS) -Isrc -I$(LLAMA)/include -I$(LLAMA)/ggml/include -I$(LLAMA)/common -I$(LLAMA)/vendor \
	  tools/hyper_server.cpp $(OBJ_ALL) -o $@ \
	  -Xlinker --start-group $(COMMON_LIBS) $(LLAMA_LIBS) -Xlinker --end-group $(LIBS) -lssl -lcrypto

$(BUILD)/hyper4: $(OBJ4) tools/hyper4_main.cpp
	$(NVCC) $(NVFLAGS) -Isrc $(GGML_INC) -Xlinker --export-dynamic tools/hyper4_main.cpp $(OBJ4) -o $@ \
	  -Xlinker --start-group $(LLAMA_LIBS) -Xlinker --end-group $(LIBS)
