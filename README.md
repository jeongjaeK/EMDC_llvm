# SYCL compiler for dynamic rebinding

This project is for providing transparency and dynamic rebinding of the use of heterogeneous accelerators on edge micro data centers 

## Ready for dynamic rebinding ##
Compile sycl code for multiple target backends with linker option to specify file offset of clang offload section

e.g. clang++ -fsycl -fsycl-targets=[backend list] main.cpp -Xlink [linker arg] (-Xlink [linker arg]) -o out.exe
  
[backends list] : comma seperated list  
  
    spir64_x86_64 for Intel CPU backend  
    spir64_gen for Intel GPU (ComputeCpp) backend  
    spir64_fpga for Intel FPGA backend(emulation or real device)  
    nvptx64 for NVIDIA PTX backend* (llvm should be configured *cuda enabled and compiled to use this target)  
  
[linker arg] : argument of GNU linker 'ld'  
  
    --section-start <section_name>=<address>  
    <section_name> : __CLANG_OFFLOAD_BUNDLE__sycl-{target backend}    
    [address] should be greater than any other section's address
 
## Work-in-progress ##
Implementing SYCL ELF strip tool.

0. parse arguments 
1. validate if the input,sycl ELF, is ready for dynamic rebinding
    file offsets of clang offload sections should be greater than other sections will be loaded into memory. 
3. dump contents of not selected clang offload sections.
  --dump-section <other clang sections>
4. empty the selected section(s)
5. save stripped sycl elf 
  
    
    
## Forked from ##
Intel LLVM-based projects (https://github.com/intel/llvm)

## README.md from Intel LLVM-based projects as below:
## oneAPI Data Parallel C++ compiler

[![](https://spec.oneapi.io/oneapi-logo-white-scaled.jpg)](https://www.oneapi.io/)

See [sycl](https://github.com/intel/llvm/tree/sycl) branch and
[DPC++ Documentation](https://intel.github.io/llvm-docs/).

[![Linux Post Commit Checks](https://github.com/intel/llvm/workflows/Linux%20Post%20Commit%20Checks/badge.svg)](https://github.com/intel/llvm/actions?query=workflow%3A%22Linux+Post+Commit+Checks%22)
[![Generate Doxygen documentation](https://github.com/intel/llvm/workflows/Generate%20Doxygen%20documentation/badge.svg)](https://github.com/intel/llvm/actions?query=workflow%3A%22Generate+Doxygen+documentation%22)

DPC++ is an open, cross-architecture language built upon the ISO C++ and Khronos
SYCL\* standards. DPC++ extends these standards with a number of extensions,
which can be found in [sycl/doc/extensions](sycl/doc/extensions) directory.

## Late-outline OpenMP\* and OpenMP\* Offload
See [openmp](https://github.com/intel/llvm/tree/openmp) branch.

# License

See [LICENSE.txt](sycl/LICENSE.TXT) for details.

# Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

*\*Other names and brands may be claimed as the property of others.*
