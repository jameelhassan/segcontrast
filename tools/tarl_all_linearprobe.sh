ckpt_dir=$1
gpu=$2
echo "${ckpt_dir}"
echo "GPU ${gpu}"

CUDA_VISIBLE_DEVICES=${gpu} python3 downstream_train.py --use-cuda --use-intensity --checkpoint tarl --contrastive --load-checkpoint --ckpt-name ${ckpt_dir} --batch-size 2 --linear-eval --sparse-model MinkUNet --epochs 50 --percentage-labels 0.01 
CUDA_VISIBLE_DEVICES=7 python3 downstream_train.py --use-cuda --use-intensity --checkpoint tarl --contrastive --load-checkpoint --ckpt-name 23_2tarl8dino_optPCD25%_BS16_1TF0dino4096_epoch=109_loss=1.55_model_t --batch-size 2 --linear-eval --sparse-model MinkUNet --epochs 50 --percentage-labels 0.01 
 
CUDA_VISIBLE_DEVICES=${gpu} python3 downstream_train.py --use-cuda --use-intensity --checkpoint tarl --contrastive --load-checkpoint --ckpt-name ${ckpt_dir} --batch-size 2 --linear-eval --sparse-model MinkUNet --epochs 100 --percentage-labels 0.001

# CUDA_VISIBLE_DEVICES=${gpu} python3 downstream_train.py --use-cuda --use-intensity --checkpoint tarl --contrastive --load-checkpoint --ckpt-name ${ckpt_dir} --batch-size 2 --linear-eval --sparse-model MinkUNet --epochs 20 --percentage-labels 0.5 

# CUDA_VISIBLE_DEVICES=${gpu} python3 downstream_train.py --use-cuda --use-intensity --checkpoint tarl --contrastive --load-checkpoint --ckpt-name ${ckpt_dir} --batch-size 2 --linear-eval --sparse-model MinkUNet --epochs 50 --percentage-labels 0.01 
 
# CUDA_VISIBLE_DEVICES=${gpu} python3 downstream_train.py --use-cuda --use-intensity --checkpoint tarl --contrastive --load-checkpoint --ckpt-name ${ckpt_dir} --batch-size 2 --linear-eval --sparse-model MinkUNet --epochs 15 --percentage-labels 1.0
CUDA_VISIBLE_DEVICES=0 python3 downstream_train.py --use-cuda --use-intensity --checkpoint tarl --contrastive --load-checkpoint --ckpt-name t6_40ep_4gpu_bs16_lr10_t25-4_m9995_epoch=99_loss=0.95_model_t --batch-size 2 --linear-eval --sparse-model MinkUNet --epochs 15 --percentage-labels 1.0