import copy
import logging
import ndjson
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
from data import TOKEN_MAP

from tqdm import tqdm

import sentencepiece as spm

SP_PATH = "/nobackup/scratch/usr/za2514/llmstep/llmstep/python/train/open_llama_3b_v2/tokenizer.model"
EOS_ID = 2


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-1.4b-deduped")


@dataclass
class DataArguments:
    train_data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    valid_data_path: str = field(default=None, metadata={"help": "Path to the validation data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

def pad_or_truncate(ids, seqlen):
    if len(ids) > seqlen:
        ids = ids[-seqlen:]
    else: 
        ids = ids + [EOS_ID for _ in range(seqlen - len(ids))]

    return ids

class TextDataset(Dataset):
    """
    Dataset for supervised fine-tuning.

    Scheme is "input_ids, labels, attention_mask"
    
    """

    def __init__(self, data_path: str, tokenizer, seqlen):
        super(Dataset, self).__init__()
        logging.warning("Loading data...")
        with open(data_path) as f: 
            list_data_dict = ndjson.load(f) 

        logging.warning("tokenizing using sentencepiece. The huggingface LlamaTokenizer has major bugs even though it's literally a sentencepiece wrapper...")

        logging.warning("Tokenizing inputs... This may take some time...")

        texts = [x['input'] + x['output'] for x in list_data_dict]
        varlen_data_list = [tokenizer.encode_as_ids(text) for text in tqdm(texts)]
        data_list = [pad_or_truncate(ids, seqlen) for ids in varlen_data_list]

        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return torch.tensor(self.data_list[i])

def data_collator(data):
    ids = torch.stack(data)

    rolled = (ids != EOS_ID).roll(shifts=1, dims=-1)
    rolled[:, 0] = 1
    return {"input_ids": ids, "labels": ids, "attention_mask": rolled}



def make_text_data_module(tokenizer: transformers.PreTrainedTokenizer, seqlen, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = TextDataset(tokenizer=tokenizer, data_path=data_args.train_data_path, seqlen=seqlen)
    eval_dataset = TextDataset(tokenizer=tokenizer, data_path=data_args.valid_data_path, seqlen=seqlen)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = spm.SentencePieceProcessor(model_file=SP_PATH)

    data_module = make_text_data_module(tokenizer=tokenizer, data_args=data_args, seqlen=training_args.model_max_length)
    trainer = Trainer(model=model, tokenizer=AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False), args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
