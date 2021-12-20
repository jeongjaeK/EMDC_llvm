### Binder ###
The binder assembles compulsory components from FAT binary, includes all device codes for all heterogeneous accelerators(Intel CPU, Intel GPU, Intel FPGA emu, Intel FPGA hw, NVIDIA GPU, AMD GPU, and Xilinx FPGA) supported by a edge micro data center(EMDC), and those runtime and library.

## components ##
Binder ( may located in master node)
SYCL compiler
GNU BIN UTILITY
Global repository

Rebinder ( may located in every worker node)
GNU BIN UTILITY
Local repository cache


## how to use ##
objcopy (binutils)

1. update unnessary section to empty file
objcopy --update-section __CLANG_OFFLOAD_BUNDLE__sycl-spir64_fpga=/dev/null ${input} ${output}

2. strip zero padding 
strip ${output}
