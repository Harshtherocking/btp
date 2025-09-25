import torch 
import os
from torch import nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor

from utils import modify_token
from dataset import Bhili_Dataset

special_tokens = [
    "bhil_Deva", 
    "gondi_Deva",
    "mun_Deva"
]



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    "indic_indic_dist_320M_mod", 
    trust_remote_code = True
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    "indic_indic_dist_320M_mod", 
    trust_remote_code = True, 
    use_cache = False,
    # attn_implementation="flash_attention_2"
).to(DEVICE)



ds = Bhili_Dataset(tokenizer)

x, y = ds[0]
x['labels'] = y['input_ids']
x.to(DEVICE)


