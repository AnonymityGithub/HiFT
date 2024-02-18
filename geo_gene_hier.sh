export num_gpus=8
export output_dir="outputs/e2e_geo"
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --nproc_per_node=$num_gpus run_glue.py \
CUDA_VISIBLE_DEVICES=3 python run_generation.py \
--model_name_or_path gpt-neo \
--model_type geo \
--dataset_name e2e_nlg \
--do_train \
--group_by_length \
--per_device_train_batch_size 8 \
--save_strategy epoch \
--evaluation_strategy epoch \
--learning_rate 2e-4 \
--lr_scheduler_type "linear" \
--pad_to_max_length \
--model_max_length 350 \
--num_train_epochs 20 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--warmup_ratio 0.0  \
--pretraining_tp 1 \
--seed 0 \
--hier_tuning \
--weight_decay 0.04 \
--group_element $1 \
--optimizer_strategy $2 \
--load_best_model_at_end
