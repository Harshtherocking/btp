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
    

#################
# Mr Dizzy here #
#################
tribal_lang_files = {
    "bhili": "./dataset/bhili-train.csv",
    "gondi": "./dataset/gondi-train.csv",
    "mundari": "./dataset/mundari-train.csv",
    "santali": "./dataset/santali-train.csv"
}

class TribalDataset(Dataset):
    def __init__(self, tokenizer, lang_name, csv_path):
        self.tokenizer = tokenizer
        self.df = pd.read_csv(csv_path)
        self.ip = IndicProcessor(inference=True)

        self.lang_map = {
            "bhili" : "bhil_Deva",
            "gondi" : "gondi_Deva",
            "mundari" : "mun_Deva"
        }

        self.tgt_tag = self.lang_map.get(lang_name.lower(), "hin_Deva")

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        src_text = str(self.df.iloc[idx]['Hindi'])
        tgt_text = str(self.df.iloc[idx].iloc[1])

        src_text = self.ip.preprocess_batch([src_text], src_lang="hin_Deva", tgt_lang="hin_Deva")[0]
        tgt_text = self.ip.preprocess_batch([tgt_text], src_lang="hin_Deva", tgt_lang="hin_Deva")[0]

        src_tokens = self.tokenizer(
            src_text,
            truncation=True,
            max_length=128,
            padding=False
        )

        tgt_tokens = self.tokenizer(
            tgt_text,
            truncation=True,
            max_length=128,
            padding=False
        )

        return {
            "input_ids": src_tokens["input_ids"],
            "attention_mask" : src_tokens["attention_mask"],
            "labels" : tgt_tokens["input_ids"]
        }

#################
#  V2 Dataset   #
#################
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from IndicTransToolkit.processor import IndicProcessor

class TribalDatasetV2(Dataset):
    LANG_CONFIG = {
        "bhili" : {"tag" : "bhil_Deva", "path": "./dataset/bhili-train.csv"},
        "gondi": {"tag": "gondi_Deva", "path": "./dataset/gondi-train.csv"},
        "mundari": {"tag": "mun_Deva", "path": "./dataset/mundari-train.csv"}
    }

    def __init__(self, tokenizer, src_lang, tgt_lang, csv_path=None, percent=1.0):
        self.tokenizer = tokenizer
        self.ip = IndicProcessor(inference=True)
    
        if csv_path is None:
            t_key = src_lang.lower() if src_lang.lower() != "hindi" else tgt_lang.lower()
            csv_path = self.LANG_CONFIG.get(t_key, {}).get("path")

        full_df = pd.read_csv(csv_path)
        self.df = full_df.sample(frac=percent, random_state=42).reset_index(drop=True)

        self.src_col = src_lang
        self.tgt_col = tgt_lang

        self.src_tag = self.LANG_CONFIG.get(src_lang.lower(), {}).get("tag", "hin_Deva")
        self.tgt_tag = self.LANG_CONFIG.get(tgt_lang.lower(), {}).get("tag", "hin_Deva")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src_text = str(self.df.iloc[idx][self.src_col])
        tgt_text = str(self.df.iloc[idx][self.tgt_col])

        src_text = self.ip.preprocess_batch([src_text], src_lang=self.src_tag, tgt_lang=self.tgt_tag)[0]
        tgt_text = self.ip.preprocess_batch([tgt_text], src_lang=self.src_tag, tgt_lang=self.tgt_tag)[0]

        src_tokens = self.tokenizer(
            src_text, truncation=True, padding="longest", return_tensors="pt"
        )
        tgt_tokens = self.tokenizer(
            tgt_text, truncation=True, padding="longest", return_tensors="pt"
        )

        return src_tokens, tgt_tokens