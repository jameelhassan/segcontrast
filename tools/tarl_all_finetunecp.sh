ckpt_dir=$1
gpu=$2
echo "${ckpt_dir}"
echo "GPU ${gpu}"

CUDA_VISIBLE_DEVICES=${gpu} python3 downstream_train.py --use-cuda --use-intensity --checkpoint tarl --contrastive --load-checkpoint --ckpt-name ${ckpt_dir} --batch-size 2 --sparse-model MinkUNet --epochs 40 --percentage-labels 0.1
CUDA_VISIBLE_DEVICES=7 python3 downstream_train.py --use-cuda --use-intensity --checkpoint tarl --contrastive --load-checkpoint --ckpt-name 43_1tf0dino_Tdrop1-25_Sdrop6-9_Scr3-6_Tcr1-25_epoch=199_loss=2.84_model_t --batch-size 2 --sparse-model MinkUNet --epochs 40 --percentage-labels 0.1

CUDA_VISIBLE_DEVICES=6 python3 downstream_train.py --use-cuda --use-intensity --checkpoint tarl --contrastive --load-checkpoint --ckpt-name t6_40ep_4gpu_bs16_lr10_t25-4_m9995_epoch=99_loss=0.95_model_t --batch-size 2 --sparse-model MinkUNet --epochs 40 --percentage-labels 0.1 --seed 2

CUDA_VISIBLE_DEVICES=${gpu} python3 downstream_train.py --use-cuda --use-intensity --checkpoint tarl --contrastive --load-checkpoint --ckpt-name ${ckpt_dir} --batch-size 2 --sparse-model MinkUNet --epochs 15 --percentage-labels 1.0 
