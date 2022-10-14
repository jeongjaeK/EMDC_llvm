#!/bin/bash

plist=$(pidof test)

for i in $plist
do
	echo $i;
	kill $i;
done

plist=$(pidof cat)

for i in $plist
do
	echo $i;
	kill $i;
done

