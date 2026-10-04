CXX      ?= c++
CXXFLAGS ?= -std=c++20 -O3 -DNDEBUG -Wall -Wextra -Wpedantic
ARCH     := $(shell uname -m)
ifeq ($(ARCH),x86_64)
CXXFLAGS += -mavx2 -mfma
endif
PY       ?= python3
HDRS     := src/hnsw.hpp src/distance.hpp
SAN      := -std=c++20 -O1 -g -fno-omit-frame-pointer

all: test py

build/test_hnsw: tests/test_hnsw.cpp $(HDRS)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $< -o $@ -pthread

build/bench_kernels: bench/bench_kernels.cpp $(HDRS)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $< -o $@

test: build/test_hnsw
	./build/test_hnsw

asan:
	@mkdir -p build
	$(CXX) $(SAN) -fsanitize=address,undefined tests/test_hnsw.cpp -o build/asan_hnsw -pthread
	./build/asan_hnsw 4000 4

tsan:
	@mkdir -p build
	$(CXX) $(SAN) -fsanitize=thread tests/test_hnsw.cpp -o build/tsan_hnsw -pthread
	./build/tsan_hnsw 4000 8

# Python extension module (needs `pip install pybind11 numpy`).
py: $(HDRS) python/bindings.cpp
	$(CXX) $(CXXFLAGS) -shared -fPIC -fvisibility=hidden $$($(PY) -m pybind11 --includes) python/bindings.cpp \
	    -o python/vecdb$$($(PY) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))") \
	    $$($(PY) -c "import sys; print('-undefined dynamic_lookup' if sys.platform == 'darwin' else '')") -pthread

clean:
	rm -rf build python/*.so

.PHONY: all test asan tsan py clean
