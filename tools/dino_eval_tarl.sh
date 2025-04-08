ckpt_dir=$1
gpu=$2
bs_dir=$3
echo ${ckpt_dir}

echo "SEGMENT CONTRAST 0.1%"
CUDA_VISIBLE_DEVICES=${gpu} python3 inference_vis.py --dataset-name SemanticKITTI --log-dir checkpoint/downstream_task/${bs_dir}percent_0.1/${ckpt_dir}/ --data-dir ./Datasets/SemanticKITTI/ --sparse-model MinkUNet --sparse-resolution 0.05 --batch-size 4 --checkpoint tarl --use-cuda --use-intensity --best epoch299 

echo "SEGMENT CONTRAST 1%"
CUDA_VISIBLE_DEVICES=${gpu} python3 inference_vis.py --dataset-name SemanticKITTI --log-dir ./checkpoint/downstream_task/${bs_dir}percent_1.0/${ckpt_dir}/ --data-dir ./Datasets/SemanticKITTI/ --sparse-model MinkUNet --sparse-resolution 0.05 --batch-size 4 --checkpoint tarl --use-cuda --use-intensity --best epoch119  

echo "SEGMENT CONTRAST 10%"
CUDA_VISIBLE_DEVICES=${gpu} python3 inference_vis.py --dataset-name SemanticKITTI --log-dir ./checkpoint/downstream_task/${bs_dir}percent_10.0/${ckpt_dir}/ --data-dir ./Datasets/SemanticKITTI/ --sparse-model MinkUNet --sparse-resolution 0.05 --batch-size 4 --checkpoint tarl --use-cuda --use-intensity --best epoch39 

echo "SEGMENT CONTRAST 50%"
CUDA_VISIBLE_DEVICES=${gpu} python3 inference_vis.py --dataset-name SemanticKITTI --log-dir ./checkpoint/downstream_task/${bs_dir}percent_50.0/${ckpt_dir}/ --data-dir ./Datasets/SemanticKITTI/ --sparse-model MinkUNet --sparse-resolution 0.05 --batch-size 4 --checkpoint tarl --use-cuda --use-intensity --best epoch19

echo "SEGMENT CONTRAST 100%"
CUDA_VISIBLE_DEVICES=${gpu} python3 inference_vis.py --dataset-name SemanticKITTI --log-dir ./checkpoint/downstream_task/${bs_dir}percent_100.0/${ckpt_dir}/ --data-dir ./Datasets/SemanticKITTI/ --sparse-model MinkUNet --sparse-resolution 0.05 --batch-size 4 --checkpoint tarl --use-cuda --use-intensity --best epoch14
