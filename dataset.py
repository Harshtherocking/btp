import os 
import pandas as pd
import torch
from torch.utils.data import Dataset 
from utils import modify_token

from IndicTransToolkit.processor import IndicProcessor
from utils import pr

bhili_data_path = os.path.join(os.getcwd(), "dataset", "bhili-train.csv")

ip = IndicProcessor(inference = True)


class Bhili_Dataset (Dataset) :
    def __init__ (self, tokenizer, path = bhili_data_path) : 
        self.path = path 
        self.df = pd.read_csv(self.path)
        self.src_col = "Hindi"
        self.tgt_col = "Bhili"
        self.tokenizer = tokenizer

    def __len__ (self) :
        return len(self.df)

    def __getitem__ (self, idx) : 
        src_text = str(self.df.iloc[idx][self.src_col])
        tgt_text = str(self.df.iloc[idx][self.tgt_col])

        src_text = ip.preprocess_batch(
            [src_text],
            src_lang = "hin_Deva",
            tgt_lang = "hin_Deva"
        )[0]

        tgt_text = ip.preprocess_batch(
            [tgt_text],
            src_lang = "hin_Deva",
            tgt_lang = "hin_Deva"
        )[0]


        # src_text = process_text[0]
        # tgt_text = process_text[1]

        
        # pr(src_text)
        # pr(tgt_text)


        src_tokens = self.tokenizer(
            src_text,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        )
        modify_token(src_tokens, self.tokenizer, "bhil_Deva" )

        # pr(src_tokens)

        tgt_tokens = self.tokenizer(
            tgt_text,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        )

        modify_token(tgt_tokens, self.tokenizer, "bhil_Deva" )

        # remove first two tokens from tgt_tokens
        # src tgt sent -> sent
        tgt_tokens["input_ids"] = tgt_tokens["input_ids"][:, 2 :]
        tgt_tokens["attention_mask"] = tgt_tokens["attention_mask"][:, 2 :]

        # pr(tgt_tokens)

        # append EOS at the end of tgt tokens
        # try : 
        #     eos_token_id = self.tokenizer.eos_token_id
        # except :
        #     eos_token_id = 2
        #     print("eos_token_id set to default : 2")
        
        # tgt_tokens.input_ids = torch.cat(
        #     ( tgt_tokens.input_ids, torch.tensor([[eos_token_id]])),
        #     dim = 1
        # )
        

        

        return src_tokens, tgt_tokens