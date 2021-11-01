# SYCL compiler for dynamic rebinding

This project is for providing transparency and dynamic rebinding of the use of heterogeneous accelerators on edge micro data centers 

## Work-in-progress ##
Removing device code section which will not be executed from FAT binary.
objcopy --remove-section ${section_name} fatbin or --update-section ${section_name}=${dummy_section.file} fatbin doesn't lead to shrink fatbin file size.

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
