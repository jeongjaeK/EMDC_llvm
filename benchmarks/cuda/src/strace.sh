#!/bin/bash

rm cuda.strace
strace -e ioctl ./a.out >> cuda.strace 2>&1