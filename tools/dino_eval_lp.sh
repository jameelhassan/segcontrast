ckpt_dir=$1
gpu=$2
echo ${ckpt_dir}

echo "SEGMENT CONTRAST 1%"
CUDA_VISIBLE_DEVICES=${gpu} python3 inference_vis.py --dataset-name SemanticKITTI --log-dir ./checkpoint/linear_eval/bs2_percent_1.0/${ckpt_dir}/ --data-dir ./Datasets/SemanticKITTI/ --sparse-model MinkUNet --sparse-resolution 0.05 --batch-size 4 --checkpoint tarl --use-cuda --use-intensity --best epoch49  

echo "SEGMENT CONTRAST 100%"
CUDA_VISIBLE_DEVICES=${gpu} python3 inference_vis.py --dataset-name SemanticKITTI --log-dir ./checkpoint/linear_eval/bs2_percent_100.0/${ckpt_dir}/ --data-dir ./Datasets/SemanticKITTI/ --sparse-model MinkUNet --sparse-resolution 0.05 --batch-size 4 --checkpoint tarl --use-cuda --use-intensity --best epoch14

 