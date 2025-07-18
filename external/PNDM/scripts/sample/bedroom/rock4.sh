#!/bin/bash
# By default, 50k samples are drawn.
METHOD="ROCK4"
#ATTENTION!!!
EPS="1e-3"
INTRA_S=50
CODE="final-tweedie"
USE_TWEEDIE="True"
#ATTENTION!!!
DATASET="bedroom"
CONFIG="ddim_${DATASET}.yml"
MODEL="models/ddim_lsun_$DATASET.ckpt"
BENCHMARK_FOLDER="nips_vis"
BENCHMARK_ROOT="${BENCHMARK_FOLDER}/${DATASET}"
OUTPUT_FILE="${BENCHMARK_ROOT}/${METHOD}_benchmarks.txt"
STATS="inception_stats/fid_${DATASET}_train.npz"


export CUDA_VISIBLE_DEVICES=0
export INTRA_S
export EPS
export USE_TWEEDIE
# Start the benchmarking loop
for i in $(seq 8 10 48); do
    sample_speed=$i
    OUTPUT="$BENCHMARK_ROOT/${METHOD}/${sample_speed}_${INTRA_S}_${EPS}_${CODE}"
    OUTPUT_SAMPLES=$OUTPUT/samples
    OUTPUT_STATS=$OUTPUT/inception_stats.npz
    mkdir -p "$OUTPUT";
    mkdir -p "$OUTPUT_SAMPLES";
    echo "Benchmarking $METHOD with sample speed $sample_speed, s=${INTRA_S}, eps=${EPS} on $DATASET, USE_TWEEDIE=${USE_TWEEDIE}"
    python main.py --runner sample --method $METHOD --sample_speed $sample_speed --device cuda --config $CONFIG --image_path $OUTPUT_SAMPLES --model_path $MODEL;
done


















