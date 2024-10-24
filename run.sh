task=$1
mkdir -p results/${task}
mkdir -p res/${task}
gpu_num=$([ $# -gt 1 ] && echo "$2" || echo "8" )

torchrun --nproc_per_node=${gpu_num} scripts/train.py ./configs/official/AIGCcode-${task}.yaml \
	--save_folder=./results/${task} \
	--save_overwrite