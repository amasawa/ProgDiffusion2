#!/bin/sh

# Define the array of eval_programs values
eval_programs=('fid10' 'fid20' 'fid50' 'fid100')

# Loop through the array and run the Python script with each value
for eval_program in "${eval_programs[@]}"
do
    python run_ffhq128.py $eval_program
done

