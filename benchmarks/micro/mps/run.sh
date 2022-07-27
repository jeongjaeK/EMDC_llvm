#!/bin/bash
CUDA_VISIBLE_DEVICES=0 nsys nvprof --print-gpu-trace bin/2mm.exe > test1.txt &
CUDA_VISIBLE_DEVICES=0 nsys nvprof --print-gpu-trace bin/2mm.exe > test2.txt &
CUDA_VISIBLE_DEVICES=0 nsys nvprof --print-gpu-trace bin/2mm.exe > test3.txt &
CUDA_VISIBLE_DEVICES=0 nsys nvprof --print-gpu-trace bin/2mm.exe > test4.txt &
CUDA_VISIBLE_DEVICES=0 nsys nvprof --print-gpu-trace bin/2mm.exe > test5.txt &
CUDA_VISIBLE_DEVICES=0 nsys nvprof --print-gpu-trace bin/2mm.exe > test6.txt &

