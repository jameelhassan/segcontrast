ckpt_dir=$1
gpu=$2
# bs_dir=$3
echo ${ckpt_dir}

echo "SEGMENT CONTRAST 0.1%"
CUDA_VISIBLE_DEVICES=${gpu} python3 inference_vis.py --dataset-name SemanticKITTI --log-dir checkpoint/downstream_task/bs2_percent_0.1/${ckpt_dir}/ --data-dir ./Datasets/SemanticKITTI/ --sparse-model MinkUNet --sparse-resolution 0.05 --batch-size 4 --checkpoint tarl --use-cuda --best epoch299 --use-intensity

echo "SEGMENT CONTRAST 1%"
CUDA_VISIBLE_DEVICES=${gpu} python3 inference_vis.py --dataset-name SemanticKITTI --log-dir ./checkpoint/downstream_task/percent_1.0/${ckpt_dir}/ --data-dir ./Datasets/SemanticKITTI/ --sparse-model MinkUNet --sparse-resolution 0.05 --batch-size 4 --checkpoint tarl --use-cuda --best epoch119 --use-intensity 

# echo "SEGMENT CONTRAST 10%"
# CUDA_VISIBLE_DEVICES=${gpu} python3 inference_vis.py --dataset-name SemanticKITTI --log-dir ./checkpoint/downstream_task/${bs_dir}percent_10.0/${ckpt_dir}/ --data-dir ./Datasets/SemanticKITTI/ --sparse-model MinkUNet --sparse-resolution 0.05 --batch-size 4 --checkpoint tarl --use-cuda --best epoch39 --use-intensity

# echo "SEGMENT CONTRAST 50%"
# CUDA_VISIBLE_DEVICES=7 python3 inference_vis.py --dataset-name SemanticKITTI --log-dir ./checkpoint/downstream_task/percent_50.0/${ckpt_dir}/ --data-dir ./Datasets/SemanticKITTI/ --sparse-model MinkUNet --sparse-resolution 0.05 --batch-size 1 --checkpoint tarl --use-cuda --best epoch19 --use-intensity

# echo "SEGMENT CONTRAST 100%"
# CUDA_VISIBLE_DEVICES=1 python3 inference_vis.py --dataset-name SemanticKITTI --log-dir ./checkpoint/downstream_task/percent_100.0/${ckpt_dir}/ --data-dir ./Datasets/SemanticKITTI/ --sparse-model MinkUNet --sparse-resolution 0.05 --batch-size 1 --checkpoint tarl --use-cuda --best epoch14 --use-intensity