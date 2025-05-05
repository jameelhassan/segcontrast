ckpt_dir=$1
gpu=$2
echo "${ckpt_dir}"
echo "GPU ${gpu}"

CUDA_VISIBLE_DEVICES=${gpu} python3 downstream_train.py --use-cuda --use-intensity --checkpoint tarl --contrastive --load-checkpoint --ckpt-name ${ckpt_dir} --batch-size 2 --sparse-model MinkUNet --epochs 120 --percentage-labels 0.01 
CUDA_VISIBLE_DEVICES=7 python3 downstream_train.py --use-cuda --use-intensity --checkpoint tarl --contrastive --load-checkpoint --ckpt-name 43_1tf0dino_Tdrop1-25_Sdrop6-9_Scr3-6_Tcr1-25_epoch=199_loss=2.84_model_t --batch-size 2 --sparse-model MinkUNet --epochs 120 --percentage-labels 0.01
 
# CUDA_VISIBLE_DEVICES=${gpu} python3 downstream_train.py --use-cuda --use-intensity --checkpoint tarl --contrastive --load-checkpoint --ckpt-name ${ckpt_dir} --batch-size 2 --sparse-model MinkUNet --epochs 300 --percentage-labels 0.001

CUDA_VISIBLE_DEVICES=${gpu} python3 downstream_train.py --use-cuda --use-intensity --checkpoint tarl --contrastive --load-checkpoint --ckpt-name ${ckpt_dir} --batch-size 2 --sparse-model MinkUNet --epochs 20 --percentage-labels 0.5 
