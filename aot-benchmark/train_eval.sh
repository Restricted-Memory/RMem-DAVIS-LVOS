exp="default"
# exp="pe_bs16_10wstep"
exp="pe_bs8_10wstep"
# exp="baseline_bs16_10wstep"
# exp="baseline_bs8_10wstep"
# exp="pe_bs8_1kstep"
gpu_num="1"
devices="0"

# model="aott"
# model="aots"
# model="aotb"
# model="aotl"
model="r50_aotl"
model="r50_deaotl"
# model="swinb_aotl"
echo "model=${model}"

## Training ##
stage="pre"
# python tools/train.py --amp \
# 	--exp_name ${exp} \
# 	--stage ${stage} \
# 	--model ${model} \
# 	--gpu_num ${gpu_num}
	
if [[ $model == "r50_aotl" ]]
then
  ckpt="R50_AOTL_PRE"
elif [[ $model == "r50_deaotl" ]]
then
  ckpt="R50_DeAOTL_PRE"
fi
echo "ckpt=${ckpt}"
ckpt_file="./pretrain_models/${ckpt}.pth"


stage="pre_ytb_dav"
result_path=$(python -c "from tools.get_config import get_config ;cfg = get_config('$stage', '$exp', '$model') ;print(cfg.DIR_RESULT)")
echo "result_path=$result_path"
# CUDA_VISIBLE_DEVICES=${devices} python tools/train.py --amp \
# 	--exp_name ${exp} \
# 	--stage ${stage} \
# 	--model ${model} \
# 	--gpu_num ${gpu_num} \
# 	--lr 2e-4 \
# 	--total_step 50000 \
# 	--batch_size 8 \
# 	--pretrained_path ${ckpt_file} \
#     --use_temporal_pos_emb \


if [[ $model == "r50_aotl" ]]
then
  ckpt="R50_AOTL_PRE_YTB_DAV"
elif [[ $model == "r50_deaotl" ]]
then
  ckpt="R50_DeAOTL_temp_pe_10wstep_80000"
fi
echo $model
echo $ckpt
ckpt_file="./pretrain_models/${ckpt}.pth"

eval_name="debug"
# eval_name="mem_1_7_nearest-flip_drop_layer_012_focus_mov_mean_0.8_ucb_reset_0_plus_8_mul_1"
# eval_name="mem_1_7_nearest-flip_drop_layer_0_focus_mov_mean_0.8_ucb_reset_0_plus_8_mul_2"
eval_name="mem_1_7_nearest-flip_drop_layer_0_focus_mov_mean_0.8_ucb_reset_0_plus_8_mul_1.5"
# eval_name="mem_1_7_nearest-flip_drop_layer_012_focus_mov_mean_0.8_ucb_reset_0_plus_8_mul_0.4"
# eval_name="mem_1_11_drop_layer_0_focus_mov_mean_0.8_ucb_reset_0_plus_8_mul_0.1"

## Evaluation ##
dataset="davis2017"
# dataset="youtubevos2018"
# dataset="youtubevos2019"
# dataset="long_videos"
# dataset="lvos"
split="val"
CUDA_VISIBLE_DEVICES=${devices} python tools/eval.py --exp_name ${exp} --stage ${stage} --model ${model} \
	--dataset ${dataset} --split ${split} --gpu_num ${gpu_num} \
	--eval_name ${eval_name} \
	--ema \
	--ckpt_path ${ckpt_file} \
	# --ckpt_step 80000 \


ema=1

if [[ $ema ]]
then
  ema="_ema"
else
  ema=""
fi

result_path="../aot-benchmark/${result_path}/eval/${dataset}/ckpt_unknown${ema}/${eval_name}"
echo "result_path=$result_path"

if [[ $dataset == "lvos" ]]
then
	cd ../lvos-evaluation/
	python evaluation_method.py \
		--task semi-supervised \
		--results_path ${result_path}/Annotations \
		--re

else
	cd ../davis2017-evaluation/
	python evaluation_method.py \
		--task semi-supervised \
		--results_path ${result_path}/Annotations/480p \
		--re

fi
