export num_gpus=8
port=$(shuf -i25000-30000 -n1)
export output_dir="outputs/e2e_gptl"
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --nproc_per_node=$num_gpus run_glue.py \
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --master_port "$port" run_generation.py \
--model_name_or_path gpt2-l \
--model_type gpt2 \
--dataset_name e2e_nlg \
--do_train \
--do_eval \
--padding_side "right" \
--group_by_length \
--deepspeed "dsconfig/zero0_config.json" \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--save_strategy epoch \
--evaluation_strategy epoch \
--learning_rate 5e-5 \
--lr_scheduler_type "linear" \
--pad_to_max_length \
--max_eval_samples 2000 \
--model_max_length 512 \
--num_train_epochs 20 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--warmup_ratio 0.01  \
--pretraining_tp 1 \
--seed 0 \
--fp16 \
--hier_tuning \
--weight_decay 0.0 \
--group_element $1 \
--optimizer_strategy $2 \
--load_best_model_at_end \
--save_total_limit 1
