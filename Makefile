# hyper: inference engine for 3x RTX 3090 + Threadripper (sm_86, AVX2)
NVCC     ?= /usr/local/cuda-12.9/bin/nvcc
CXX      ?= g++
LLAMA    ?= $(HOME)/llama.cpp-src
ARCH     ?= -gencode arch=compute_86,code=sm_86   # 4th GPU (Blackwell): make ARCH="... -gencode arch=compute_120,code=sm_120"
CXXFLAGS := -O3 -march=native -std=c++17 -Wall -Wno-unused-function -fopenmp
NVFLAGS  := -O3 -std=c++17 $(ARCH) -lineinfo --use_fast_math -Xcompiler "-O3 -march=native -fopenmp" $(EXTRA)
LIBS     := -L/usr/local/cuda-12.9/lib64 -lcudart -lcublas -lcuda -lgomp -lpthread -ldl

BUILD := build
# engine objects (CPU experts through ggml)
OBJ4  := $(BUILD)/gguf.o $(BUILD)/dense.o $(BUILD)/experts.o $(BUILD)/moe_place.o $(BUILD)/engine_base.o $(BUILD)/tp_engine.o $(BUILD)/model4.o $(BUILD)/kernels.o $(BUILD)/kernels4.o $(BUILD)/cpu_moe.o $(BUILD)/engine4.o \
         $(BUILD)/model5.o $(BUILD)/kernels5.o $(BUILD)/engine5.o \
         $(BUILD)/model7.o $(BUILD)/kernels7.o $(BUILD)/engine7.o
GGML_INC := -I$(LLAMA)/ggml/include -I$(LLAMA)/ggml/src

all: $(BUILD)/hyper-server $(BUILD)/hyper4 $(BUILD)/hyper7 $(BUILD)/ref $(BUILD)/arbench

$(BUILD)/%.o: src/%.cpp src/*.h
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) $(GGML_INC) -x cu -c $< -o $@

$(BUILD)/%.o: src/%.cu src/*.h src/*.cuh
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) $(GGML_INC) -c $< -o $@

# deepseek41 kernels: no fast math (rotary angles up to ~262k radians, the reference's F32 softmaxes and roundings)
$(BUILD)/kernels7.o: src/kernels7.cu src/kernels7.cuh
	@mkdir -p $(BUILD)
	$(NVCC) $(filter-out --use_fast_math,$(NVFLAGS)) $(GGML_INC) -c $< -o $@

# golden reference against mainline llama.cpp (static libs from $(LLAMA)/build)
LLAMA_LIBS := $(LLAMA)/build/src/libllama.a $(LLAMA)/build/ggml/src/ggml-cuda/libggml-cuda.a \
              $(LLAMA)/build/ggml/src/libggml-cpu.a $(LLAMA)/build/ggml/src/libggml.a $(LLAMA)/build/ggml/src/libggml-base.a
$(BUILD)/ref: tools/ref.cpp
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -I$(LLAMA)/include -I$(LLAMA)/ggml/include tools/ref.cpp -o $@ \
	  -Wl,--start-group $(LLAMA_LIBS) -Wl,--end-group $(LIBS)

# tokenizer only (offline experiments on request dumps)
$(BUILD)/tokdump: tools/tokdump.cpp
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -I$(LLAMA)/include -I$(LLAMA)/ggml/include tools/tokdump.cpp -o $@ \
	  -Wl,--start-group $(LLAMA_LIBS) -Wl,--end-group $(LIBS)

clean:
	rm -rf $(BUILD)

.PHONY: all clean

$(BUILD)/arbench: tools/arbench.cu $(BUILD)/kernels.o
	$(NVCC) $(NVFLAGS) tools/arbench.cu $(BUILD)/kernels.o -o $@ $(LIBS)

# Flash-Next QSA decode path (indexer selection + sparse attention) at depth, synthetic caches
$(BUILD)/qsabench: tools/qsabench.cu $(BUILD)/kernels.o $(BUILD)/kernels4.o $(BUILD)/kernels5.o $(BUILD)/gguf.o
	$(NVCC) $(NVFLAGS) -Isrc $(GGML_INC) tools/qsabench.cu $(BUILD)/kernels.o $(BUILD)/kernels4.o $(BUILD)/kernels5.o $(BUILD)/gguf.o -o $@ $(LIBS)

# Gated DeltaNet prefill step, random inputs (hashes of output and state)
$(BUILD)/gdnbench: tools/gdnbench.cu $(BUILD)/kernels.o
	$(NVCC) $(NVFLAGS) tools/gdnbench.cu $(BUILD)/kernels.o -o $@ $(LIBS)

# prefill grouped expert GEMMs (Flash-Next shapes), random blocks and routing (hashes of the outputs)
$(BUILD)/moegemmbench: tools/moegemmbench.cu $(BUILD)/kernels.o $(BUILD)/kernels4.o $(BUILD)/gguf.o
	$(NVCC) $(NVFLAGS) -Isrc $(GGML_INC) tools/moegemmbench.cu $(BUILD)/kernels.o $(BUILD)/kernels4.o $(BUILD)/gguf.o -o $@ $(LIBS)

# dense Q8 tensor-core GEMM (prefill), random weights (hash of the output)
$(BUILD)/gemmbench: tools/gemmbench.cu $(BUILD)/kernels.o
	$(NVCC) $(NVFLAGS) tools/gemmbench.cu $(BUILD)/kernels.o -o $@ $(LIBS)

# MoE routing kernel
$(BUILD)/routebench: $(OBJ4) tools/routebench.cu
	$(NVCC) $(NVFLAGS) -Isrc $(GGML_INC) tools/routebench.cu $(OBJ4) -o $@ \
	  -Xlinker --start-group $(LLAMA_LIBS) -Xlinker --end-group $(LIBS)

# MoE decode kernels (Flash-Next shapes)
$(BUILD)/moebench: $(OBJ4) tools/moebench.cu
	$(NVCC) $(NVFLAGS) -Isrc $(GGML_INC) tools/moebench.cu $(OBJ4) -o $@ \
	  -Xlinker --start-group $(LLAMA_LIBS) -Xlinker --end-group $(LIBS)

$(BUILD)/topkbench: $(OBJ4) tools/topkbench.cu
	$(NVCC) $(NVFLAGS) -Isrc $(GGML_INC) tools/topkbench.cu $(OBJ4) -o $@ \
	  -Xlinker --start-group $(LLAMA_LIBS) -Xlinker --end-group $(LIBS)

$(BUILD)/mlabench: $(OBJ4) tools/mlabench.cu
	$(NVCC) $(NVFLAGS) -Isrc $(GGML_INC) tools/mlabench.cu $(OBJ4) -o $@ \
	  -Xlinker --start-group $(LLAMA_LIBS) -Xlinker --end-group $(LIBS)

$(BUILD)/fp16gemm: tools/fp16gemm.cu
	$(NVCC) $(NVFLAGS) tools/fp16gemm.cu -o $@ $(LIBS)

$(BUILD)/kqbench: tools/kqbench.cu $(BUILD)/kernels.o $(BUILD)/gguf.o
	$(NVCC) $(NVFLAGS) -Isrc tools/kqbench.cu $(BUILD)/kernels.o $(BUILD)/gguf.o -o $@ $(LIBS)

$(BUILD)/iq3bench: tools/iq3bench.cpp src/iq3s_dot.h src/gguf.cpp
	$(CXX) $(CXXFLAGS) -Isrc $(GGML_INC) tools/iq3bench.cpp src/gguf.cpp -o $@ \
	  -Wl,--start-group $(LLAMA)/build/ggml/src/libggml-cpu.a $(LLAMA)/build/ggml/src/libggml-base.a -Wl,--end-group -ldl -lpthread

$(BUILD)/iq2bench: tools/iq2bench.cpp src/iq2_dot.h src/gguf.cpp
	$(CXX) $(CXXFLAGS) -Isrc $(GGML_INC) tools/iq2bench.cpp src/gguf.cpp -o $@ \
	  -Wl,--start-group $(LLAMA)/build/ggml/src/libggml-cpu.a $(LLAMA)/build/ggml/src/libggml-base.a -Wl,--end-group -ldl -lpthread

# CPU expert row dots per block type (ggml), real expert rows: compute cost and streaming bandwidth
$(BUILD)/iqbench: tools/iqbench.cpp src/iq1s_dot.h src/gguf.cpp
	$(CXX) $(CXXFLAGS) -Isrc $(GGML_INC) tools/iqbench.cpp src/gguf.cpp -o $@ \
	  -Wl,--start-group $(LLAMA)/build/ggml/src/libggml-cpu.a $(LLAMA)/build/ggml/src/libggml-base.a -Wl,--end-group -ldl -lpthread

$(BUILD)/cpumoebench: $(OBJ4) tools/cpumoebench.cu
	$(NVCC) $(NVFLAGS) -Isrc $(GGML_INC) tools/cpumoebench.cu $(OBJ4) -o $@ \
	  -Xlinker --start-group $(LLAMA_LIBS) -Xlinker --end-group $(LIBS)

# dense GEMV on GGUF blocks: bandwidth on the Flash-Next K-quant shapes
$(BUILD)/ggbench: $(OBJ4) tools/ggbench.cu
	$(NVCC) $(NVFLAGS) -Isrc $(GGML_INC) tools/ggbench.cu $(OBJ4) -o $@ \
	  -Xlinker --start-group $(LLAMA_LIBS) -Xlinker --end-group $(LIBS)

# expert dequantization (deq8) vs ggml's reference
$(BUILD)/samplertest: tools/samplertest.cpp src/sampling.h src/llm.h
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) tools/samplertest.cpp -o $@

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
$(BUILD)/arxbench: tools/arxbench.cu $(BUILD)/kernels.o
	$(NVCC) $(NVFLAGS) tools/arxbench.cu $(BUILD)/kernels.o -o $@ $(LIBS)

# OpenAI-compatible server: chat templates / tool-call parsing from mainline libcommon, vocab from libllama
COMMON_LIBS := $(LLAMA)/build/common/libllama-common.a $(LLAMA)/build/common/libllama-common-base.a \
               $(LLAMA)/build/vendor/cpp-httplib/libcpp-httplib.a \
               $(LLAMA)/build/tools/mtmd/libmtmd.a $(LLAMA)/build/vendor/hash/libvendor-hash.a
OBJ_ALL := $(OBJ4)
$(BUILD)/hyper-server: $(OBJ_ALL) tools/hyper_server.cpp tools/chat_page.h
	$(NVCC) $(NVFLAGS) -Isrc -I$(LLAMA)/include -I$(LLAMA)/ggml/include -I$(LLAMA)/common -I$(LLAMA)/vendor -I$(LLAMA)/tools/mtmd \
	  tools/hyper_server.cpp $(OBJ_ALL) -o $@ \
	  -Xlinker --start-group $(COMMON_LIBS) $(LLAMA_LIBS) -Xlinker --end-group $(LIBS) -lssl -lcrypto

$(BUILD)/hyper7: $(OBJ4) tools/hyper7_main.cpp
	$(NVCC) $(NVFLAGS) -Isrc $(GGML_INC) tools/hyper7_main.cpp $(OBJ4) -o $@ \
	  -Xlinker --start-group $(LLAMA_LIBS) -Xlinker --end-group $(LIBS)

$(BUILD)/hyper4: $(OBJ4) tools/hyper4_main.cpp
	$(NVCC) $(NVFLAGS) -Isrc $(GGML_INC) -Xlinker --export-dynamic tools/hyper4_main.cpp $(OBJ4) -o $@ \
	  -Xlinker --start-group $(LLAMA_LIBS) -Xlinker --end-group $(LIBS)

$(BUILD)/g32bench: tools/g32bench.cu $(BUILD)/kernels.o $(BUILD)/kernels4.o $(BUILD)/gguf.o
	$(NVCC) $(NVFLAGS) -Isrc $(GGML_INC) tools/g32bench.cu $(BUILD)/kernels.o $(BUILD)/kernels4.o $(BUILD)/gguf.o -o $@ $(LIBS)

# grouped GEMV / row-tile views vs a CPU reference (DeepSeek attn_output_a)
$(BUILD)/grouptest: tools/grouptest.cu $(BUILD)/kernels.o $(BUILD)/kernels4.o $(BUILD)/gguf.o $(BUILD)/dense.o
	$(NVCC) $(NVFLAGS) -Isrc $(GGML_INC) tools/grouptest.cu $(BUILD)/kernels.o $(BUILD)/kernels4.o $(BUILD)/gguf.o $(BUILD)/dense.o -o $@ \
	  -Xlinker --start-group $(LLAMA_LIBS) -Xlinker --end-group $(LIBS)

$(BUILD)/experts.o: src/experts.cu src/experts.h src/devices.h src/cuda_check.h src/cpu_moe.h src/gguf.h
	$(NVCC) $(NVFLAGS) -Isrc $(GGML_INC) -c src/experts.cu -o $@

$(BUILD)/moe_place.o: src/moe_place.cu src/moe_place.h src/experts.h src/devices.h src/cuda_check.h src/cpu_moe.h src/vmm.h src/kernels4.cuh
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) $(GGML_INC) -c $< -o $@

$(BUILD)/tp_engine.o: src/tp_engine.cu src/tp_engine.h src/moe_place.h src/experts.h src/devices.h src/cuda_check.h src/cpu_moe.h src/vmm.h src/park.h src/kernels4.cuh src/llm.h src/sampling.h src/dense.h
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) $(GGML_INC) -c $< -o $@

$(BUILD)/engine_base.o: src/engine_base.cu src/engine_base.h src/moe_place.h src/devices.h src/cuda_check.h src/cpu_moe.h src/llm.h
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) $(GGML_INC) -c $< -o $@
