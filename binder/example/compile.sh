#!/bin/bash

SRC=scan.cpp
OUTPUT=scan_fat.exe
EMPTY_EXE=scan
TRT_DEV=spir64_x86_64,spir64_gen,spir64_fpga,nvptx64_nvidia
#x86_64 : Intel CPU, gen : Intel GPU, fpga : Intel FPGA, nvptx64 : NVIDIA GPU

### Compile sycl code for multiple target devices with specified section start addresses ###
clang++ -fsycl -fsycl-targets=$TRT_DEV \
	-Xlinker --section-start -Xlinker __CLANG_OFFLOAD_BUNDLE__sycl-spir64_x86_64=0x5000000 \
	-Xlinker --section-start -Xlinker __CLANG_OFFLOAD_BUNDLE__sycl-spir64_gen=0x6000000 \
	-Xlinker --section-start -Xlinker __CLANG_OFFLOAD_BUNDLE__sycl-spir64_fpga=0x7000000 \
	-Xlinker --section-start -Xlinker __CLANG_OFFLOAD_BUNDLE__sycl-nvptx64-nvidia-cuda=0x8000000 \
	-z max-page-size=4096 \
	$SRC -o $OUTPUT

### for validation ###
echo "Fat binary ..."
readelf -SWl $OUTPUT

### dump each section ###
objcopy --dump-section __CLANG_OFFLOAD_BUNDLE__sycl-spir64_x86_64=spir64_x86_64.sec $OUTPUT
objcopy --dump-section __CLANG_OFFLOAD_BUNDLE__sycl-spir64_gen=spir64_gen.sec $OUTPUT
objcopy --dump-section __CLANG_OFFLOAD_BUNDLE__sycl-spir64_fpga=spir64_fpga.sec $OUTPUT
objcopy --dump-section __CLANG_OFFLOAD_BUNDLE__sycl-nvptx64=nvptx64-nvidia.sec $OUTPUT

### strip ###
strip $OUTPUT

### update all clang offloaded section for binding ###
objcopy --update-section __CLANG_OFFLOAD_BUNDLE__sycl-spir64_x86_64=/dev/null $OUTPUT $EMPTY_EXE
objcopy --update-section __CLANG_OFFLOAD_BUNDLE__sycl-spir64_gen=/dev/null $EMPTY_EXE
objcopy --update-section __CLANG_OFFLOAD_BUNDLE__sycl-spir64_fpga=/dev/null $EMPTY_EXE
objcopy --update-section __CLANG_OFFLOAD_BUNDLE__sycl-nvptx64-nvidia=/dev/null $EMPTY_EXE

### binding ###
objcopy --update-section __CLANG_OFFLOAD_BUNDLE__sycl-spir64_x86_64=spir64_x86_64.sec $EMPTY_EXE ${EMPTY_EXE}_x86.exe
objcopy --update-section __CLANG_OFFLOAD_BUNDLE__sycl-spir64_gen=spir64_gen.sec $EMPTY_EXE ${EMPTY_EXE}_gen.exe
objcopy --update-section __CLANG_OFFLOAD_BUNDLE__sycl-spir64_fpga=spir64_fpga.sec $EMPTY_EXE ${EMPTY_EXE}_fpga.exe
objcopy --update-section __CLANG_OFFLOAD_BUNDLE__sycl-nvptx64-nvidia=nvptx64-nvidia.sec $EMPTY_EXE ${EMPTY_EXE}_nvidia.exe

### execution ###
SYCL_DEVICE_FILTER=cpu ./${EMPTY_EXE}_x86.exe
SYCL_DEVICE_FILTER=gpu ./${EMPTY_EXE}_gen.exe
SYCL_DEVICE_FILTER=acc ./${EMPTY_EXE}_fpga.exe
SYCL_DEVICE_FILTER=gpu ./${EMPTY_EXE}_nvidia.exe

### for comparision ###
ls -l --block-size=K
readelf -SW ${EMPTY_EXE}_x86.exe
readelf -SW ${EMPTY_EXE}_gen.exe
readelf -SW ${EMPTY_EXE}_fpga.exe
readelf -SW ${EMPTY_EXE}_nvidia.exe

### rebining from intel cpu to intel fpga###
SYCL_DEVICE_FILTER=cpu ./${EMPTY_EXE}_x86.exe
objcopy --update-section __CLANG_OFFLOAD_BUNDLE__sycl-spir64_x86_64=/dev/null ${EMPTY_EXE}_x86.exe rebind.exe
objcopy --update-section __CLANG_OFFLOAD_BUNDLE__sycl-spir64_fpga=spir64_fpga.sec rebind.exe
