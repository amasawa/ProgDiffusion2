#!/bin/bash

# Define the array of eval_programs values
eval_programs=(
  'checkpoints/ffhq128_Nopre_10ckpt/epoch=31-step=17000.ckpt'
  'checkpoints/ffhq128_Nopre_10ckpt/epoch=62-step=34000.ckpt'
  'checkpoints/ffhq128_Nopre_10ckpt/epoch=93-step=51000.ckpt'
  'checkpoints/ffhq128_Nopre_10ckpt/epoch=124-step=68000.ckpt'
  'checkpoints/ffhq128_Nopre_10ckpt/epoch=155-step=85000.ckpt'
  'checkpoints/ffhq128_Nopre_10ckpt/epoch=186-step=102000.ckpt'
  'checkpoints/ffhq128_Nopre_10ckpt/epoch=217-step=119000.ckpt'
  'checkpoints/ffhq128_Nopre_10ckpt/epoch=249-step=136000.ckpt'
  'checkpoints/ffhq128_Nopre_10ckpt/epoch=280-step=153000.ckpt'
  'checkpoints/ffhq128_Nopre_10ckpt/epoch=311-step=170000.ckpt'
)

# Loop through the array and run the Python script with each value
for eval_program in "${eval_programs[@]}"
do
    python run_ffhq128.py "$eval_program"
done
