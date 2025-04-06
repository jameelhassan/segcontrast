
# echo "SEGMENT CONTRAST 10%"
# CUDA_VISIBLE_DEVICES=6 python3 inference_vis.py --dataset-name SemanticKITTI --log-dir ./checkpoint/downstream_task/percent_10.0/DINO_means_BS4_LR6_t2-5_epoch=09_loss=1.20_model_t/ --data-dir ./Datasets/SemanticKITTI/ --sparse-model MinkUNet --sparse-resolution 0.05 --batch-size 1 --checkpoint tarl --use-cuda --best epoch39 --use-intensity & wait;\
# CUDA_VISIBLE_DEVICES=6 python3 inference_vis.py --dataset-name SemanticKITTI --log-dir ./checkpoint/downstream_task/percent_10.0/DINO_means10_BS4_lr2_t3-4_epoch=39_loss=2.33_model_t/ --data-dir ./Datasets/SemanticKITTI/ --sparse-model MinkUNet --sparse-resolution 0.05 --batch-size 1 --checkpoint tarl --use-cuda --best epoch39 --use-intensity & wait;\
# CUDA_VISIBLE_DEVICES=6 python3 inference_vis.py --dataset-name SemanticKITTI --log-dir ./checkpoint/downstream_task/percent_10.0/DINO_means30ep_BS4_LR2_t3-4.5_epoch=09_loss=1.94_model_t/ --data-dir ./Datasets/SemanticKITTI/ --sparse-model MinkUNet --sparse-resolution 0.05 --batch-size 1 --checkpoint tarl --use-cuda --best epoch39 --use-intensity & wait;\
# CUDA_VISIBLE_DEVICES=6 python3 inference_vis.py --dataset-name SemanticKITTI --log-dir ./checkpoint/downstream_task/percent_10.0/DINO_means50ep_BS4_LR4_t3-5_epoch=09_loss=1.42_model_t/ --data-dir ./Datasets/SemanticKITTI/ --sparse-model MinkUNet --sparse-resolution 0.05 --batch-size 1 --checkpoint tarl --use-cuda --best epoch39 --use-intensity & wait;\

BASE_DIR="./checkpoint/downstream_task/percent_10.0"

for dir in "$BASE_DIR"/*; do
    if [ -d "$dir" ]; then
        DIR_NAME=$(basename "$dir")
        echo "Running inference for checkpoint: $DIR_NAME"
        CUDA_VISIBLE_DEVICES=2 python3 inference_vis.py \
            --dataset-name SemanticKITTI \
            --log-dir "$BASE_DIR/$DIR_NAME/" \
            --data-dir ./Datasets/SemanticKITTI/ \
            --sparse-model MinkUNet \
            --sparse-resolution 0.05 \
            --batch-size 1 \
            --checkpoint tarl \
            --use-cuda \
            --best epoch39 \
            --use-intensity
    fi
done