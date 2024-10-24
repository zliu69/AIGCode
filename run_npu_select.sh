task=$1
npu_indice=$2
mkdir -p results/${task}
mkdir -p res/${task}
gpu_num=$([ $# -gt 1 ] && echo "$3" || echo "8" )
# torchrun --nproc_per_node=1 scripts/train.py ./configs/task/AIGCode-${task}.yaml 
# torchrun 
# python -m torch.distributed.run 
export NPU_VISIBLE_DEVICES=${npu_indice}
export ASCEND_RT_VISIBLE_DEVICES=${npu_indice}
torchrun --nproc_per_node=${gpu_num} scripts/train.py ./configs/official/AIGCcode-${task}.yaml \
	--save_folder=./results/${task} \
	--save_overwrite