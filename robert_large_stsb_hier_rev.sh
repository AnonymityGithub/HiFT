export num_gpus=8
export output_dir="outputs/stsb_o"
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --nproc_per_node=$num_gpus run_glue.py \
--model_name_or_path roberta-large \
--task_name stsb \
--do_train \
--do_eval \
--do_predict \
--max_seq_length 512 \
--per_device_train_batch_size 4 \
--learning_rate 2e-5 \
--num_train_epochs 50 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--evaluation_strategy epoch \
--save_strategy epoch \
--warmup_ratio 0 \
--seed 0 \
--hier_tuning \
--hier_reverse \
--weight_decay 0 \
--group_element $1 \
--optimizer_strategy $2 \
--load_best_model_at_end
