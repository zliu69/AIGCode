#!/usr/bin/env bash

set -ex

NUM_NODES=8

gantry run \
  --workspace ai2/AIGCcode-training \
  --task-name mitchish70-from160510 \
  --description "AIGCcode large - 70B - from160510" \
  --priority normal \
  --beaker-image shanea/aigcode-torch2.2-gantry \
  --cluster ai2/jupiter-cirrascale \
  --cluster ai2/pluto-cirrascale \
  --preemptible \
  --gpus 8 \
  --replicas "${NUM_NODES}" \
  --leader-selection \
  --host-networking \
  --budget ai2/oe-training \
  --no-nfs \
  --propagate-failure \
  --synchronized-start-timeout 10m \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env AIGCODE_TASK=model \
  --env-secret WANDB_API_KEY=DIRKG_WANDB_API_KEY \
  --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \
  --env-secret AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY \
  --shared-memory 10GiB \
  --venv base \
  --yes \
  --timeout=-1 \
  -- /bin/bash -c "scripts/beaker/mitchish70-from160510.sh \$BEAKER_LEADER_REPLICA_HOSTNAME ${NUM_NODES} \$BEAKER_REPLICA_RANK"
