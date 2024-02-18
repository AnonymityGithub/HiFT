export num_gpus=8
export output_dir="outputs/o_e2e_gpt2xl"
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --nproc_per_node=$num_gpus run_glue.py \
CUDA_VISIBLE_DEVICES=3 python run_generation.py \
--model_name_or_path gpt2-l \
--model_type gpt2 \
--dataset_name e2e_nlg \
--do_train \
--do_eval \
--padding_side "left" \
--group_by_length \
--predict_with_generate \
--per_device_train_batch_size 16 \
--save_strategy "steps" \
--evaluation_strategy "steps" \
--max_eval_samples 2000 \
--learning_rate 5e-5 \
--lr_scheduler_type "linear" \
--pad_to_max_length \
--model_max_length 512 \
--num_train_epochs 20 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--warmup_ratio 0.0  \
--pretraining_tp 1 \
--seed 0 \
--weight_decay 0.04 \
--load_best_model_at_end
