# echo "SEGMENT CONTRAST 0.1%"
# python3 inference_vis.py --dataset-name SemanticKITTI --log-dir ./checkpoint/downstream_task/TARL/ --data-dir ./Datasets/SemanticKITTI/ --sparse-model MinkUNet --sparse-resolution 0.05 --batch-size 1 --checkpoint segment_contrast_0p001 --use-cuda --best epoch299 --use-intensity & wait;\

# echo "SEGMENT CONTRAST 1%"
# python3 inference_vis.py --dataset-name SemanticKITTI --log-dir ./checkpoint/downstream_task/TARL/ --data-dir ./Datasets/SemanticKITTI/ --sparse-model MinkUNet --sparse-resolution 0.05 --batch-size 1 --checkpoint segment_contrast_0p01 --use-cuda --best epoch119 --use-intensity & wait;\

# echo "SEGMENT CONTRAST 10%"
# python3 inference_vis.py --dataset-name SemanticKITTI --log-dir ./checkpoint/downstream_task/TARL/ --data-dir ./Datasets/SemanticKITTI/ --sparse-model MinkUNet --sparse-resolution 0.05 --batch-size 1 --checkpoint segment_contrast_0p1 --use-cuda --best epoch39 --use-intensity & wait;\

# echo "SEGMENT CONTRAST 50%"
# python3 inference_vis.py --dataset-name SemanticKITTI --log-dir ./checkpoint/downstream_task/TARL/ --data-dir ./Datasets/SemanticKITTI/ --sparse-model MinkUNet --sparse-resolution 0.05 --batch-size 1 --checkpoint segment_contrast_0p5 --use-cuda --best epoch19 --use-intensity & wait;\

# echo "SEGMENT CONTRAST 100%"
# python3 inference_vis.py --dataset-name SemanticKITTI --log-dir ./checkpoint/downstream_task/TARL/ --data-dir ./Datasets/SemanticKITTI/ --sparse-model MinkUNet --sparse-resolution 0.05 --batch-size 1 --checkpoint segment_contrast_1p0 --use-cuda --best epoch14 --use-intensity & wait;\

# echo "SEGMENT CONTRAST 1%"
# python3 inference_vis.py --dataset-name SemanticKITTI --log-dir ./checkpoint/downstream_task/percent2_1.0/ --data-dir ./Datasets/SemanticKITTI/ --sparse-model MinkUNet --sparse-resolution 0.05 --batch-size 1 --checkpoint tarl --use-cuda --best epoch119 --use-intensity & wait;\

#!/bin/bash

BASE_DIR="./checkpoint/downstream_task/percent_1.0"

for dir in "$BASE_DIR"/*; do
    if [ -d "$dir" ]; then
        DIR_NAME=$(basename "$dir")
        echo "Running inference for checkpoint: $DIR_NAME"
        CUDA_VISIBLE_DEVICES=1 python3 inference_vis.py \
            --dataset-name SemanticKITTI \
            --log-dir "$BASE_DIR/$DIR_NAME/" \
            --data-dir ./Datasets/SemanticKITTI/ \
            --sparse-model MinkUNet \
            --sparse-resolution 0.05 \
            --batch-size 1 \
            --checkpoint tarl \
            --use-cuda \
            --best epoch119 \
            --use-intensity
    fi
done