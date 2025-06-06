#!/bin/bash
#SBATCH --job-name=aigcode-small-ablation
#SBATCH --account=project_462000229
#SBATCH --output=/pfs/lustref1/flash/project_462000229/logs/%j.log
#SBATCH --nodes=16              # Total number of nodes
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8       # Allocate one gpu per MPI rank
#SBATCH --cpus-per-task=6
#SBATCH --time=48:00:00
#SBATCH --mem=0			# All memory on the node
#SBATCH --partition=standard-g

module load LUMI/22.08 partition/G

# check if LOAD_PATH is provided as an environment variable; if so, create an argument
# to pass to the training script
if [ -z ${LOAD_PATH+x} ]; then
  LOAD_PATH_ARG=""
else
  LOAD_PATH_ARG="--load_path=${LOAD_PATH}"
fi


export AIGCODE_CONTAINER=llm-lumi_latest.sif

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=3
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1
export FI_CXI_DISABLE_CQ_HUGETLB=1

# We need to set this to avoid "Cassini Event Queue overflow detected." errors.
export FI_CXI_DEFAULT_CQ_SIZE=131072

#export NCCL_DEBUG=INFO
export PYTHONPATH=.:${PYTHONPATH}
export WANDB_PROJECT=c4-small
export ROCM_PATH=/opt/rocm
export SINGULARITYENV_LD_LIBRARY_PATH=/usr/local/lib:/opt/cray/libfabric/1.15.2.0/lib64
export CONFIG_PATH=configs/aigcode-small-ablation.yaml

# Try playing with max_split_size_mb if you run into OOM errors.
# export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512

# get run name, we will postpend it with the job id of this slurm run
export RUN_NAME=$(cat $CONFIG_PATH | grep -ohP "^run_name\:\w*(.+)$" | sed 's/run_name:\s*//')

# actually run the training script
srun \
  --cpus-per-task=$SLURM_CPUS_PER_TASK \
  --distribution=block:block \
  --kill-on-bad-exit \
  scripts/run_with_environment.sh \
    singularity exec \
    -B"$PROJECT_DIR:$PROJECT_DIR" \
    -B"$SCRATCH_DIR:$SCRATCH_DIR" \
    -B"$FLASH_DIR:$FLASH_DIR" \
    -B /opt/cray:/opt/cray \
    -B /usr/lib64/libcxi.so.1:/usr/lib64/libcxi.so.1 \
    -B /usr/lib64/libjson-c.so.3:/usr/lib64/libjson-c.so.3 \
    $PROJECT_DIR/containers/$AIGCODE_CONTAINER \
    python scripts/train.py $CONFIG_PATH \
      --run_name="${RUN_NAME}_${SLURM_JOB_ID}" \
      --wandb.project=$WANDB_PROJECT \
      $LOAD_PATH_ARG \
      ${@}
