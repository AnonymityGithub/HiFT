# HiFT

## Results without prompts

1. Hierarchical Fine-Tuning RoEBRTa-base  on CoLA  

script "robert_base_cola_hier.sh"  

```
export num_gpus=8
export output_dir="outputs/cola"
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --nproc_per_node=$num_gpus run_glue.py \
--model_name_or_path roberta-base \
--task_name cola \
--do_train \
--do_eval \
--do_predict \
--max_seq_length 512 \
--per_device_train_batch_size 8 \
--learning_rate 3e-5 \
--num_train_epochs 1 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--evaluation_strategy epoch \
--save_strategy epoch \
--warmup_ratio 0.02 \
--seed 0 \
--hier_tuning \
--weight_decay 0 \
--group_element $1 \
--optimizer_strategy $2 \
--load_best_model_at_end \
--metric_for_best_model "matthews_correlation"
```

#### Run command "bash robert_base_cola_hier.sh 1 down2up "  

2. Hierarchical Fine-Tuning RoEBRTa-large  on CoLA  

script "robert_large_cola_hier.sh"  

```
export num_gpus=8
export output_dir="outputs/cola"
port=$(shuf -i25000-30000 -n1)
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --nproc_per_node=$num_gpus --master_port "$port" run_glue.py \
--model_name_or_path roberta-large \
--task_name cola \
--do_train \
--do_eval \
--do_predict \
--max_seq_length 512 \
--per_device_train_batch_size 4 \
--learning_rate 3e-5 \
--num_train_epochs 1 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--evaluation_strategy epoch \
--save_strategy epoch \
--warmup_ratio 0.01 \
--seed 0 \
--hier_tuning \
--weight_decay 0 \
--group_element $1 \
--optimizer_strategy $2 \
--load_best_model_at_end

```

#### Run command "bash robert_large_cola_hier.sh 1 down2up "  



## Results with prompts

The source code in prompts_code path is based on MeZO.  

1. If you want to get the results of RoEBRTa-large, run the script “finetune.sh" under “prompts_code/medium_models”.  

   

2. If you want to get the results of opt-13B, run the script “finetune.sh" under “prompts_code/large_models”.











