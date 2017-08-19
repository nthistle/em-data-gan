#!/bin/sh

export CUDA_VISIBLE_DEVICES=$1
python run_model.py $2 $3 $4 $5 $6 $7 $8 $9 $10 $11 $12 > "output/em_gan_$1_$3_$4_$5_$8_$9_$10_$11_$12.out" 2>&1

