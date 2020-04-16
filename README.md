MAD filter GPU : CUDA-based RFI mitigation algorithm using Median Absolute Deviation (MAD).

There are two versions of the code: `madfilter_small` and `madfilter_large`

To compile the code use the following command:
`nvcc -Xptxas="-v" -o madfilter_small madfilter_small.cu -arch=sm_??`

Make sure you have `help_small.sh` in the same directory as the compiled binary to generate a help message

`madfilter_small` runs a MAD based filter on SIGPROC format filterbank data. Each GPU thread runs a filter on all time samples of a single frequency channel. The output is stored in a binary file with the same base name as the input file and `_filtered` appended to it. The output file does not have the original SIGPROC header so that needs to be added to the output to make it compatible with standard pulsar processing software. An example of generating a cleaned filterbank file is the following:

`./madfilter_small 3000000 672 3 -m 256 find_the_pulse.fil`;
`head -c 256 find_the_pulse.fil > header.txt`;
`cat header.txt find_the_pulse.fil_filtered > find_the_pulse_filtered.fil`
