#!/bin/bash

echo "Help for MAD filter:"
echo "usage: ./madfilter_small <number of samples> <window size> 
<threshold> <filtering option> <size of header in bytes>"

echo "Options for filtering are :"
echo "-z :  zero padding"
echo "-r: replace with random integer based on noise rms"
echo "-c: replace with threshold"
echo "-m: replace with median"
