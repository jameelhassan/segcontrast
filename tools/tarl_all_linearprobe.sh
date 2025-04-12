ckpt_dir=$1
gpu=$2
echo "${ckpt_dir}"
echo "GPU ${gpu}"

CUDA_VISIBLE_DEVICES=${gpu} python3 downstream_train.py --use-cuda --use-intensity --checkpoint tarl --contrastive --load-checkpoint --ckpt-name ${ckpt_dir} --batch-size 2 --linear-eval --sparse-model MinkUNet --epochs 50 --percentage-labels 0.01 
 
CUDA_VISIBLE_DEVICES=${gpu} python3 downstream_train.py --use-cuda --use-intensity --checkpoint tarl --contrastive --load-checkpoint --ckpt-name ${ckpt_dir} --batch-size 2 --linear-eval --sparse-model MinkUNet --epochs 15 --percentage-labels 1.0