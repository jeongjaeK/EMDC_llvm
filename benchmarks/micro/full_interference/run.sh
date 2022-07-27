#!/bin/bash

mkdir -p 32

./a.out &
for i in {1..32}
do
	nsys nvprof ./a.out $i > ./32/$i.txt &
done
