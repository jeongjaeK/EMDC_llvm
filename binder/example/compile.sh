#! /bin/bash

clang++ -fsycl -fsycl-targets=spir64_x86_64,spir64_fpga scan.cpp -o scan
