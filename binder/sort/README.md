### locate device code section ###

## Using linker option ##

To place the section as close end of file offset, specify start address of the section as greater than other sections' address.

e.g. ld --section-start=__CLANG_OFFLOAD_BUNDLE__sycl-spir64_x86_64=0x8000000 ...

How to combine with clang++ compier...

clang++ ... -Xlinker <arg> -Xlinker <arg>  # for multiple sections


