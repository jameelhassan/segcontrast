ckpt_dir=$1
gpu=$2
bs_dir=$3
echo ${ckpt_dir}

echo "SEGMENT CONTRAST 0.1%"
CUDA_VISIBLE_DEVICES=${gpu} python3 inference_vis.py --dataset-name SemanticKITTI --log-dir checkpoint/downstream_task/${bs_dir}percent_0.1/${ckpt_dir}/ --data-dir ./Datasets/SemanticKITTI/ --sparse-model MinkUNet --sparse-resolution 0.05 --batch-size 4 --checkpoint tarl --use-cuda --use-intensity --best epoch299 

echo "SEGMENT CONTRAST 1%"
CUDA_VISIBLE_DEVICES=${gpu} python3 inference_vis.py --dataset-name SemanticKITTI --log-dir ./checkpoint/downstream_task/${bs_dir}percent_1.0/${ckpt_dir}/ --data-dir ./Datasets/SemanticKITTI/ --sparse-model MinkUNet --sparse-resolution 0.05 --batch-size 4 --checkpoint tarl --use-cuda --use-intensity --best epoch119  
# FT
CUDA_VISIBLE_DEVICES=7 python3 inference_vis.py --dataset-name SemanticKITTI --log-dir checkpoint/downstream_task/seed0_percent_1.0/43_1tf0dino_Tdrop1-25_Sdrop6-9_Scr3-6_Tcr1-25_epoch=199_loss=2.84_model_t --data-dir ./Datasets/SemanticKITTI/ --sparse-model MinkUNet --sparse-resolution 0.05 --batch-size 1 --checkpoint tarl --use-cuda --use-intensity --best epoch119  
# LP
CUDA_VISIBLE_DEVICES=5 python3 inference_vis.py --dataset-name SemanticKITTI --log-dir checkpoint/linear_eval/seed0_percent_1.0/52_2xTeachpts_8tarl2dino_BS16_LR8_25%_Tdrop1-25_Sdrop6-9_Tcr1-25_Scr3-6_T3_epoch=109_loss=1.04_model_t --data-dir ./Datasets/SemanticKITTI/ --sparse-model MinkUNet --sparse-resolution 0.05 --batch-size 1 --checkpoint tarl --use-cuda --best epoch49 --use-intensity

echo "SEGMENT CONTRAST 10%"
CUDA_VISIBLE_DEVICES=${gpu} python3 inference_vis.py --dataset-name SemanticKITTI --log-dir ./checkpoint/downstream_task/${bs_dir}percent_10.0/${ckpt_dir}/ --data-dir ./Datasets/SemanticKITTI/ --sparse-model MinkUNet --sparse-resolution 0.05 --batch-size 4 --checkpoint tarl --use-cuda --use-intensity --best epoch39 
CUDA_VISIBLE_DEVICES=7 python3 inference_vis.py --dataset-name SemanticKITTI --log-dir checkpoint/downstream_task/seed0_percent_10.0/43_1tf0dino_Tdrop1-25_Sdrop6-9_Scr3-6_Tcr1-25_epoch=199_loss=2.84_model_t --data-dir ./Datasets/SemanticKITTI/ --sparse-model MinkUNet --sparse-resolution 0.05 --batch-size 1 --checkpoint tarl --use-cuda --use-intensity --best epoch39  


echo "SEGMENT CONTRAST 50%"
CUDA_VISIBLE_DEVICES=${gpu} python3 inference_vis.py --dataset-name SemanticKITTI --log-dir ./checkpoint/downstream_task/${bs_dir}percent_50.0/${ckpt_dir}/ --data-dir ./Datasets/SemanticKITTI/ --sparse-model MinkUNet --sparse-resolution 0.05 --batch-size 4 --checkpoint tarl --use-cuda --use-intensity --best epoch19

echo "SEGMENT CONTRAST 100%"
CUDA_VISIBLE_DEVICES=${gpu} python3 inference_vis.py --dataset-name SemanticKITTI --log-dir ./checkpoint/downstream_task/${bs_dir}percent_100.0/${ckpt_dir}/ --data-dir ./Datasets/SemanticKITTI/ --sparse-model MinkUNet --sparse-resolution 0.05 --batch-size 4 --checkpoint tarl --use-cuda --use-intensity --best epoch14

#LP
CUDA_VISIBLE_DEVICES=9 python3 inference_vis.py --dataset-name SemanticKITTI --log-dir checkpoint/linear_eval/bs2_percent_100.0/DINO40ep_t6_4gpu_BS16_lr6_t3-4.5_epoch=59_loss=1.25_model_t --data-dir ./Datasets/SemanticKITTI/ --sparse-model MinkUNet --sparse-resolution 0.05 --batch-size 1 --checkpoint tarl --use-cuda --use-intensity --best epoch14

#FT
CUDA_VISIBLE_DEVICES=7 python3 inference_vis.py --dataset-name SemanticKITTI --log-dir checkpoint/downstream_task/bs2_percent_100.0/DINO20ep_t6_4gpu_BS16_lr6_epoch=179_loss=1.37_model_t --data-dir ./Datasets/SemanticKITTI/ --sparse-model MinkUNet --sparse-resolution 0.05 --batch-size 1 --checkpoint tarl --use-cuda --use-intensity --best epoch14
