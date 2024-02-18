export num_gpus=1
export output_dir="outputs/sst2"
port=$(shuf -i25000-30000 -n1)
CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --nproc_per_node=$num_gpus --master_port "$port" run_glue.py \
--model_name_or_path roberta-large \
--task_name sst2 \
--do_train \
--do_eval \
--do_predict \
--max_seq_length 512 \
--per_device_train_batch_size 16 \
--learning_rate 1e-5 \
--num_train_epochs 1 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--evaluation_strategy epoch \
--save_strategy epoch \
--warmup_ratio 0 \
--seed 0 \
--weight_decay 0 \
--load_best_model_at_end
