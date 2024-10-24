
task=$1
# mkdir -p results/${task}
# mkdir -p res/${task}
mkdir -p results/${task}
mkdir -p res/${task}

gpu_num=$([ $# -gt 1 ] && echo "$2" || echo "8" )
task=$1
# mkdir -p results/${task}
# mkdir -p res/${task}
mkdir -p results/${task}
mkdir -p res/${task}

gpu_num=$([ $# -gt 1 ] && echo "$2" || echo "8" )

export MASTER_ADDR=10.0.14.44
export MASTER_PORT=12340
export NODE_NUM=8
export NODE_RANK=5
# source env_npu.sh
# 关闭HCCL通道白名单
export HCCL_WHITELIST_DISABLE=1
# HCCL初始化通信网卡IP，设置为当前服务器的host IP
# export HCCL_IF_IP=10.0.0.176
export HCCL_IF_IP=10.0.18.68
# 
torchrun --nproc_per_node=${gpu_num} \
 --nnodes=$NODE_NUM \
 --node_rank=$NODE_RANK  \
 --master_addr=$MASTER_ADDR \
 --master_port=$MASTER_PORT scripts/train.py ./configs/official/AIGCcode-${task}.yaml \
 --save_folder=./results/${task} \
 --save_overwrite


# export MASTER_ADDR=10.181.132.236
# export MASTER_ADDR=10.0.0.176
# export MASTER_PORT=12340
# export NODE_NUM=2
# export NODE_RANK=1

# # export TORCH_DISTRIBUTED_DEBUG=INFO
# export NCCL_IB_DISABLE=1
# # export NCCL_DEBUG=INFO
# # export NCCL_DEBUG_SUBSYS=ALL
# # export TORCH_DISTRIBUTED_DEBUG=INFO

# NCCL_SOCKET_IFNAME=eth0 NCCL_CROSS_NIC=1 torchrun --nproc_per_node=${gpu_num} \
#  --local_addr=10.181.132.46 \
#  --nnodes=$NODE_NUM \
#  --node_rank=$NODE_RANK  \
#  --master_addr=$MASTER_ADDR \
#  --master_port=$MASTER_PORT scripts/train.py ./configs/task/AIGCode-${task}.yaml \
#  --save_folder=./results/${task} \
#  --save_overwrite

# torchrun --nproc_per_node=1 scripts/train.py ./configs/task/AIGCode-${task}.yaml 
# torchrun --nproc_per_node=${gpu_num} scripts/train.py ./configs/task/AIGCode-${task}.yaml \
# 	--save_folder=./results/${task} \
# 	--save_overwrite
