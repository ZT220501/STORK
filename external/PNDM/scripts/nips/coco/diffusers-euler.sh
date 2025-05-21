#!/bin/bash
# By default, 20k samples are drawn.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

METHOD="flow_euler"
MODEL="stabilityai/stable-diffusion-3.5-medium"
NUM_SAMPLES=30000
NUM_PROC=8
SEED=0
PRECISION="bfloat16"
OVERWRITE="True"
IMAGE_SIZE=512
CFG_SCALE=3.5


BENCHMARK_FOLDER="nips_data"
DATASET="coco-30k_512"
BENCHMARK_ROOT="${BENCHMARK_FOLDER}/${MODEL}/${DATASET}"
OUTPUT_FILE="${BENCHMARK_ROOT}/${METHOD}_${INTRA_S}_${PRECISION}_cfg${CFG_SCALE}_benchmarks.txt"
STATS="inception_stats/${DATASET}.npz"

# Start the benchmarking loop
for i in $(seq 8 8 40); do
    sample_speed=$i
    OUTPUT="$BENCHMARK_ROOT/${METHOD}/${sample_speed}_numims${NUM_SAMPLES}_${PRECISION}_cfg${CFG_SCALE}"
    OUTPUT_SAMPLES=$OUTPUT/samples
    OUTPUT_STATS=$OUTPUT/inception_stats.npz
    mkdir -p "$OUTPUT";
    echo "Benchmarking $METHOD with sample speed $sample_speed on $DATASET"
    python -m stable_diffusion \
        --model_id $MODEL \
        --num_samples $NUM_SAMPLES \
        --num_inference_steps $sample_speed \
        --scheduler $METHOD \
        --num_proc $NUM_PROC\
        --save_dir $OUTPUT_SAMPLES \
        --seed $SEED \
        --precision $PRECISION \
        --overwrite $OVERWRITE \
        --image_size $IMAGE_SIZE \
        --cfg_scale $CFG_SCALE \
        --dataset $DATASET
    # calculate FID
    # Pre-compute the FID statistics
    python -m pytorch_fid --save-stats $OUTPUT_SAMPLES $OUTPUT_STATS
    # Use the pre-computed statistics to calculate FID
    echo "DATASET=$STATS, RUN=$OUTPUT_STATS" >> "$OUTPUT_FILE"
    python -m pytorch_fid $STATS $OUTPUT_STATS | grep "FID:" >> "$OUTPUT_FILE"
done



















