export num_gpus=8
export output_dir="outputs/e2e_llama"
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --nproc_per_node=$num_gpus run_generation.py \
CUDA_VISIBLE_DEVICES=0 python run_generation.py \
--model_name_or_path llama2-7b \
--model_type llama \
--dataset_name e2e_nlg \
--do_train \
--deepspeed "dsconfig/zero3_config.json" \
--group_by_length \
--per_device_train_batch_size 1 \
--save_strategy epoch \
--model_max_length 512 \
--evaluation_strategy epoch \
--set_fp16 \
--fp16 \
--learning_rate 8e-6 \
--lr_scheduler_type "linear" \
--pad_to_max_length \
--num_train_epochs 20 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--warmup_ratio 0.0  \
--pretraining_tp 1 \
--seed 0 \
--weight_decay 0.04 \
--group_element $1 \
--optimizer_strategy $2 \
--load_best_model_at_end
