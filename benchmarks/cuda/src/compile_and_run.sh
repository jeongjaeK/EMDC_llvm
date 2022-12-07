#!/bin/bash

sudo dmesg -c
rm cuda.dmesg
nvcc -arch=sm_80 -DDEBUG runtime_apis.cu && CUDA_VISIBLE_DEVICES=0,1 ./a.out && dmesg > cuda.dmesg; 
code cuda.dmesg
