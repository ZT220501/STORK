#!/bin/bash
# By default, 50k samples are drawn.
METHOD="DEIS"
DATASET="cifar10"
CONFIG="ddim_${DATASET}.yml"
MODEL="models/ddim_$DATASET.ckpt"  
BENCHMARK_FOLDER="nips_vis"
BENCHMARK_ROOT="${BENCHMARK_FOLDER}/${DATASET}"
OUTPUT_FILE="${BENCHMARK_ROOT}/${METHOD}_benchmarks.txt"
STATS="inception_stats/fid_${DATASET}_train.npz"


export CUDA_VISIBLE_DEVICES=0
# Start the benchmarking loop
for i in $(seq 10 10 100); do
    sample_speed=$i
    OUTPUT="$BENCHMARK_ROOT/${METHOD}/${sample_speed}"
    OUTPUT_SAMPLES=$OUTPUT/samples
    OUTPUT_STATS=$OUTPUT/inception_stats.npz
    mkdir -p "$OUTPUT";
    echo "Benchmarking $METHOD with sample speed $sample_speed on $DATASET"
    python diffuser_wrapper.py --runner sample --method $METHOD --sample_speed $sample_speed --device cuda --config $CONFIG --image_path $OUTPUT_SAMPLES --model_path $MODEL;
done


















