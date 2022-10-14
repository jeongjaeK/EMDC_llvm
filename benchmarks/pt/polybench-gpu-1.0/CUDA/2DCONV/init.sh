#!/bin/bash

for i in $(seq 1 $1)
do
	./test < fifo.$i &
done


#./test < my.fifo2 &
#./test < my.fifo3 &
