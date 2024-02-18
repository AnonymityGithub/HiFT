export output_dir="outputs/e2e_gptl"
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --nproc_per_node=$num_gpus run_glue.py \
CUDA_VISIBLE_DEVICES=1 python inference.py \
--model_name_or_path gpt2-l \
--model_type gpt2 \
--dataset_name e2e_nlg \
--do_predict \
--per_device_eval_batch_size 1 \
--max_input_length 170 \
--max_target_length 350 \
--pad_to_max_length \
--checkpoints_path $output_dir/model \
--seed 0 \
--temperature 0.8 \
--k 1 \
--p 0.8 \
--num_beams 10 \
--do_sample \
--num_return_sequences 1 \
--repetition_penalty 4 \
--length_penalty 0.9
