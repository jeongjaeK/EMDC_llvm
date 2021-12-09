### locate device code section ###

## Using linker option ##

To place the section as close end of file offset, specify start address of the section as greater than other sections' address.

e.g. ld --section-start=__CLANG_OFFLOAD_BUNDLE__sycl-spir64_x86_64=0x8000000 ...

How to combine with clang++ compier...

clang++ ... -Xlinker <arg> -Xlinker <arg>  # for multiple sections

## dump offloaded sections ##

For dynamic rebinding, dump CLANG offloaded sections using BINUTILITY such as objcopy or llvm-objcopy

e.g. objcopy --dump-section __CLANG_OFFLOAD_BUNDLE__sycl-spir64_x86_64=[out-file] [in-file]

## update unnecessary offloaded sections to null ##

To reduce fat binary size, update unnecessary CLANG offloaded sections to empty section using objcopy.

(removing sections make difficult to rebind.)

e.g. objcopy --update-section __CLANG_OFFLOAD_BUNDLE__sycl-spir64_x86_64=/dev/null [in-file]

