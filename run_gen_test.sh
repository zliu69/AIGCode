
npu_indice=$1
gen_type=$2
export NPU_VISIBLE_DEVICES=${npu_indice}
export ASCEND_RT_VISIBLE_DEVICES=${npu_indice}
python  scripts/generaion_from_ckpt_test.py \
    --npu_indce ${npu_indice} \
    --gen_type ${gen_type}