#!/usr/bin/env python
# coding=utf-8
import os
import logging
import json
from typing import Optional,Tuple
import argparse
import datasets
import evaluate
import nltk
from nltk import word_tokenize
import numpy as np
from datasets import load_dataset
from filelock import FileLock
from datasets import load_dataset
from glob import glob
import torch
from accelerate.utils import set_seed
from llama.modeling_llama import LlamaForCausalLM
from llama.tokenization_llama import LlamaTokenizer
from llama.configuration_llama import LlamaConfig
import transformers
from functools import partial
from transformers import GenerationConfig

from torch.utils.data import Dataset,RandomSampler, SequentialSampler,DataLoader

from transformers import (
    AutoConfig,
    GPT2Config,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    BloomForCausalLM,
    BloomTokenizerFast,
    BloomConfig,
    CTRLLMHeadModel,
    CTRLTokenizer,
    CTRLConfig,
    GenerationMixin,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPTJForCausalLM,
    GPTJConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    OpenAIGPTConfig,
    OPTForCausalLM,
    OPTConfig,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    TransfoXLConfig,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLMConfig,
    XLNetLMHeadModel,
    XLNetTokenizer,
    XLNetConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.trainer_utils import get_last_checkpoint


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer,GPT2Config),
    "geo":(AutoModelForCausalLM,AutoTokenizer,AutoConfig),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer,CTRLConfig),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,OpenAIGPTConfig),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer,XLNetConfig),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer,TransfoXLConfig),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer,XLMConfig),
    "gptj": (GPTJForCausalLM, AutoTokenizer,GPTJConfig),
    "bloom": (BloomForCausalLM, BloomTokenizerFast,BloomConfig),
    "llama": (LlamaForCausalLM, LlamaTokenizer,LlamaConfig),
    "opt": (OPTForCausalLM, GPT2Tokenizer,OPTConfig),
}
datasets_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
    "multi_news": ("document", "summary"),
    "e2e_nlg":("meaning_representation", "human_reference"),
}
class Features(object):
    def __init__(self, src,target):
        self.src = src
        self.target = target
class DatasetWraper(Dataset):
    def __init__(self, examples):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]
def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def prepare_ctrl_input(args, _, tokenizer, prompt_text):
    if args.temperature > 0.7:
        logger.info("CTRL typically works better with lower temperatures (and lower top_k).")

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
    return prompt_text

def prepare_xlm_input(args, model, tokenizer, prompt_text):
    # kwargs = {"language": None, "mask_token_id": None}

    # Set the language
    use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
    if hasattr(model.config, "lang2id") and use_lang_emb:
        available_languages = model.config.lang2id.keys()
        if args.xlm_language in available_languages:
            language = args.xlm_language
        else:
            language = None
            while language not in available_languages:
                language = input("Using XLM. Select language in " + str(list(available_languages)) + " >>> ")

        model.config.lang_id = model.config.lang2id[language]
        # kwargs["language"] = tokenizer.lang2id[language]

    # TODO fix mask_token_id setup when configurations will be synchronized between models and tokenizers
    # XLM masked-language modeling (MLM) models need masked token
    # is_xlm_mlm = "mlm" in args.model_name_or_path
    # if is_xlm_mlm:
    #     kwargs["mask_token_id"] = tokenizer.mask_token_id

    return prompt_text
def prepare_xlnet_input(args, _, tokenizer, prompt_text):
    prefix = args.prefix if args.prefix else args.padding_text if args.padding_text else PREFIX
    prompt_text = prefix + prompt_text
    return prompt_text


def prepare_transfoxl_input(args, _, tokenizer, prompt_text):
    prefix = args.prefix if args.prefix else args.padding_text if args.padding_text else PREFIX
    prompt_text = prefix + prompt_text
    return prompt_text

def preprocess_function_for_generation(config,examples):
    new_examples = []
    SEPTOKEN=" ### "
    for i in range(len(examples[config.source_column])):
        if examples[config.source_column][i] and examples[config.target_column][i]:
            src = examples[config.source_column][i].strip().rstrip()+SEPTOKEN
            tgt = examples[config.target_column][i].strip().rstrip()
            new_examples.append(Features(src=src,target = tgt))
            
    return new_examples
PREPROCESSING_FUNCTIONS = {
    "gpt2":preprocess_function_for_generation,
    "geo":preprocess_function_for_generation,
    "llama":preprocess_function_for_generation,
}

def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length

def sparse_model_config(model_config):
    embedding_size = None
    if hasattr(model_config, "hidden_size"):
        embedding_size = model_config.hidden_size
    elif hasattr(model_config, "n_embed"):
        embedding_size = model_config.n_embed
    elif hasattr(model_config, "n_embd"):
        embedding_size = model_config.n_embd

    num_head = None
    if hasattr(model_config, "num_attention_heads"):
        num_head = model_config.num_attention_heads
    elif hasattr(model_config, "n_head"):
        num_head = model_config.n_head

    if embedding_size is None or num_head is None or num_head == 0:
        raise ValueError("Check the model config")

    num_embedding_size_per_head = int(embedding_size / num_head)
    if hasattr(model_config, "n_layer"):
        num_layer = model_config.n_layer
    elif hasattr(model_config, "num_hidden_layers"):
        num_layer = model_config.num_hidden_layers
    else:
        raise ValueError("Number of hidden layers couldn't be determined from the model config")

    return num_layer, num_head, num_embedding_size_per_head

def generate_past_key_values(model, batch_size, seq_len):
    num_block_layers, num_attention_heads, num_embedding_size_per_head = sparse_model_config(model.config)
    if model.config.model_type == "bloom":
        past_key_values = tuple(
            (
                torch.empty(int(num_attention_heads * batch_size), num_embedding_size_per_head, seq_len)
                .to(model.dtype)
                .to(model.device),
                torch.empty(int(num_attention_heads * batch_size), seq_len, num_embedding_size_per_head)
                .to(model.dtype)
                .to(model.device),
            )
            for _ in range(num_block_layers)
        )
    else:
        past_key_values = tuple(
            (
                torch.empty(batch_size, num_attention_heads, seq_len, num_embedding_size_per_head)
                .to(model.dtype)
                .to(model.device),
                torch.empty(batch_size, num_attention_heads, seq_len, num_embedding_size_per_head)
                .to(model.dtype)
                .to(model.device),
            )
            for _ in range(num_block_layers)
        )
    return past_key_values
def prepare_jit_inputs(inputs, model, tokenizer):
    batch_size = len(inputs)
    dummy_input = tokenizer.batch_encode_plus(inputs, return_tensors="pt")
    dummy_input = dummy_input.to(model.device)
    if model.config.use_cache:
        dummy_input["past_key_values"] = generate_past_key_values(model, batch_size, 1)
    dummy_input["attention_mask"] = torch.cat(
        [
            torch.zeros(dummy_input["attention_mask"].shape[0], 1)
            .to(dummy_input["attention_mask"].dtype)
            .to(model.device),
            dummy_input["attention_mask"],
        ],
        -1,
    )
    return dummy_input
class _ModelFallbackWrapper(GenerationMixin):
    __slots__ = ("_optimized", "_default")

    def __init__(self, optimized, default):
        self._optimized = optimized
        self._default = default

    def __call__(self, *args, **kwargs):
        if kwargs["past_key_values"] is None and self._default.config.use_cache:
            kwargs["past_key_values"] = generate_past_key_values(self._default, kwargs["input_ids"].shape[0], 0)
        kwargs.pop("position_ids", None)
        for k in list(kwargs.keys()):
            if kwargs[k] is None or isinstance(kwargs[k], bool):
                kwargs.pop(k)
        outputs = self._optimized(**kwargs)
        lm_logits = outputs[0]
        past_key_values = outputs[1]
        fixed_output = CausalLMOutputWithPast(
            loss=None,
            logits=lm_logits,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )
        return fixed_output

    def __getattr__(self, item):
        return getattr(self._default, item)

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs
    ):
        return self._default.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs
        )

    def _reorder_cache(
        self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PretrainedModel.beam_search`] or
        [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return self._default._reorder_cache(past_key_values, beam_idx)
def get_test_dataloader(args,test_dataset,tokenizer):
    def collate_fn(batch_data,tokenizer,max_input_length,max_target_length,padding):
        input_string = []
        target_string = []
        for d in batch_data:
            input_string.append(d.src)
            target_string.append(d.target)
        
        model_inputs = tokenizer(input_string,
            return_tensors="pt",
            max_length=max_input_length,
            padding=padding,
            truncation=True)
        labels = tokenizer(target_string,
                return_tensors="pt",
                max_length=max_target_length, 
                padding=padding, 
                truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    def create_collate_fn(tokenizer, max_input_length, max_target_length,padding):
        f = partial(collate_fn, 
                    tokenizer=tokenizer, 
                    max_input_length=max_input_length, 
                    max_target_length=max_target_length,
                    padding=padding)
        return f
    FeaturesDataset = DatasetWraper(test_dataset)
    test_sampler = SequentialSampler(FeaturesDataset)
    padding = "max_length" if args.pad_to_max_length else False
    _collate_fn = create_collate_fn(tokenizer = tokenizer,
                                    max_input_length=args.max_input_length,
                                    max_target_length =args.max_target_length,
                                    padding = False)
    return DataLoader(
            FeaturesDataset,
            sampler=test_sampler,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=_collate_fn,
            drop_last=False
        )
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--checkpoints_path",
        default=None,
        type=str,
        required=True,
        help="......"
    )
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--max_input_length",type=int,default=170)
    parser.add_argument("--max_target_length",type=int,default=350)
    parser.add_argument("--preprocessing_num_workers",type=int,default=4)
    parser.add_argument("--overwrite_cache",action="store_true",help="...")
    parser.add_argument("--pad_to_max_length",action="store_true",help="...")
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--length_penalty",type=float,default=1.0)
    parser.add_argument("--dataset_name",type=str,default=None,help="the dataset name")
    parser.add_argument("--dataset_config_name",type=str,default=None,help="")
    parser.add_argument("--test_file",type=str,default=None,help="the file for inference")
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument("--num_beams",type=int,default=1)
    parser.add_argument("--per_device_eval_batch_size",type=int,default=8)
    parser.add_argument("--prefix", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--padding_text", type=str, default="", help="Deprecated, the use of `--prefix` is preferred.")
    parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")
    parser.add_argument("--do_sample",action="store_true",help="...")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Whether or not to use cpu. If set to False, " "we will use gpu/npu or mps device if available",
    )
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument("--jit", action="store_true", help="Whether or not to use jit trace to accelerate inference")
    parser.add_argument("--do_eval",action="store_true", help="inference on dev datasets")
    parser.add_argument("--do_predict",action="store_true", help="inference on test datasets")
    parser.add_argument("--source_column",type=str,default=None,help="the source column name")
    parser.add_argument("--target_column",type=str,default=None,help="the target column name")
    args = parser.parse_args()

    # Initialize the distributed state.
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.warning(f"device: {args.device}, 16-bits inference: {args.fp16}")

    if args.seed is not None:
        set_seed(args)
    assert args.dataset_name is not None
    # Downloading and loading a dataset from the hub.
    raw_datasets = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        cache_dir=None,
        )
    if args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        column_names = raw_datasets["validation"].column_names
    elif args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return
    dataset_columns = datasets_name_mapping.get(args.dataset_name, None)
    if args.source_column is None:
        source_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
        args.source_column = source_column
    else:
        source_column = args.source_column
        if source_column not in column_names:
            raise ValueError(
                f"--source_column' value '{args.sourcec_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.target_column is None:
        target_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
        args.target_column = target_column
    else:
        target_column = args.target_column
        if target_column not in column_names:
            raise ValueError(
                f"--target_column' value '{args.target_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.do_eval:
        inference_dataset = raw_datasets["validation"]
    elif args.do_predict:
        inference_dataset = raw_datasets["test"]
    
    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class,model_config = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")
    assert args.model_type in PREPROCESSING_FUNCTIONS.keys()
    config = model_config.from_pretrained(args.model_name_or_path)
    generation_config = GenerationConfig.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path,padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    preprocess_function = PREPROCESSING_FUNCTIONS.get(args.model_type)
    examples = preprocess_function(config=args, examples = inference_dataset)
    test_dataloader  = get_test_dataloader(args,examples,tokenizer)
    checkpoints = glob(os.path.join(args.checkpoints_path,"checkpoint-*"))
    for checkpoint_file in checkpoints:
        model = model_class.from_pretrained(checkpoint_file,config=config)
        # Set the model to the right device
        model.to(args.device)
        if args.fp16:
            model.half()
        max_seq_length = getattr(model.config, "max_position_embeddings", 0)
        args.length = adjust_length_to_model(args.max_target_length, max_sequence_length=max_seq_length)
        logger.info(args)
        generated_sequences,references = [],[]
        for i, batch in enumerate(test_dataloader):
            if args.jit:
                jit_input_texts = ["enable jit"]
                jit_inputs = prepare_jit_inputs(jit_input_texts, model, tokenizer)
                torch._C._jit_set_texpr_fuser_enabled(False)
                model.config.return_dict = False
                if hasattr(model, "forward"):
                    sig = inspect.signature(model.forward)
                else:
                    sig = inspect.signature(model.__call__)
                jit_inputs = tuple(jit_inputs[key] for key in sig.parameters if jit_inputs.get(key, None) is not None)
                traced_model = torch.jit.trace(model, jit_inputs, strict=False)
                traced_model = torch.jit.freeze(traced_model.eval())
                traced_model(*jit_inputs)
                traced_model(*jit_inputs)

                model = _ModelFallbackWrapper(traced_model, model)
            predictions = model.generate(
                input_ids=batch["input_ids"].to(args.device),
                attention_mask = batch["attention_mask"].to(args.device),
                max_length=args.length + batch["input_ids"].size()[-1],
                temperature=args.temperature,
                generation_config = generation_config,
                top_k=args.k,
                top_p=args.p,
                num_beams=args.num_beams,
                length_penalty = args.length_penalty,
                repetition_penalty=args.repetition_penalty,
                do_sample=args.do_sample,
                num_return_sequences=args.num_return_sequences,
            )
            # Remove the batch dimension when returning multiple sequences
            if len(predictions.shape) > 2:
                predictions.squeeze_()
            input_list = batch["input_ids"].cpu().tolist()
            predictions = predictions.cpu()
            predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
            # predictions = [predict[len(input_list[example_index]):] for example_index, predict in enumerate(predictions)]
    
            predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            predictions = [pred.strip() for pred in predictions]

            batch_reference = batch["labels"].tolist()
            batch_reference = np.where(batch_reference != -100, batch_reference, tokenizer.pad_token_id)
            batch_reference = tokenizer.batch_decode(batch_reference, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            batch_reference = [ref.strip() for ref in batch_reference]
            references.extend(batch_reference)
            generated_sequences.extend(predictions)
        generated_sequences = [prediction for prediction in generated_sequences]
        references=[tgt for tgt in references]
        with open(os.path.join(checkpoint_file,"predict.json"),"w",encoding="utf-8") as f:
            json.dump(generated_sequences,f,ensure_ascii=False,indent=4)
        with open(os.path.join(checkpoint_file,"reference.json"),"w",encoding="utf-8") as f:
            json.dump(references,f,ensure_ascii=False,indent=4)
        

if __name__ == "__main__":
    main()