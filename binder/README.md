### Binder ###
The binder assembles compulsory components from FAT binary, includes all device codes for all heterogeneous accelerators supported by a EMDC, and those libraries.

## components ##
GNU Bin utility

Global repository



## how to use ##
objcopy (binutils)

1. update unnessary section to empty file
objcopy --update-section __CLANG_OFFLOAD_BUNDLE__sycl-spir64_fpga=/dev/null ${input} ${output}

2. strip zero padding 
strip ${output}
