#!/bin/bash
mkdir -p tmp
section_list=`cat ./section_list`

#objcopy --update-section __CLANG_OFFLOAD_BUNDLE__sycl-spir64_fpga=/dev/null t1

#for i in $section_list;do
#	objcopy --dump-section $i=tmp$i t1
#done
#for i in $section_list;do
#	objcopy --update-section $i=/dev/null t1
#done
#for i in $section_list;do
#	echo $i
#	objcopy --change-section-lma $i-104050 t1 t2
#done
for i in $section_list;do
	objcopy --update-section $i=tmp$i t4
done

