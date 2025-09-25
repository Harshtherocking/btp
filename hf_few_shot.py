import pandas as pd 
import os
from tqdm import tqdm
import sacrebleu

hindi_path = "/home/harsh22/btp/few_shot_hindi_tribal"
english_path = "/home/harsh22/btp/few_shot_eng_tribal"

langs = ["Hindi", "English"]
tribal_langs = ["Bhili", "Garo", "Gondi", "Kui_Odia", "Mundari", "Santali"]



def template(input, src_lang, target_lang, examples) : 
    few_shot_prompt = f"Below are some examples of {src_lang} to {target_lang} translations:\n"

    for src_sent , target_sent in examples:
        few_shot_prompt += f"{src_lang}: {src_sent}\n{target_lang}: {target_sent}\n\n"

    few_shot_prompt += f"Translate the following {src_lang} sentence to {target_lang}:\n{src_lang}: {input}\n{target_lang}:"
    return few_shot_prompt




# Use a pipeline as a high-level helper
from transformers import pipeline
pipe = pipeline("text-generation", model="google/gemma-3-270m", device_map = "auto")
print(pipe("hi", max_new_tokens = 4096, return_full_text = False))
exit()



for hrl in langs: 
    for lrl in tribal_langs : 


        # dataset path
        if hrl == "Hindi": 
            data_path = os.path.join(hindi_path, f"hindi_{lrl.lower()}.csv")
        else : 
            data_path = os.path.join(english_path, f"eng_{lrl.lower()}.csv")
        
        try :
            df = pd.read_csv(data_path)
            if hrl not in df.columns or lrl not in df.columns : 
                raise ValueError(f"Coloumns Mismatch\nHRL: {hrl}\nLRL: {lrl}\nFound: {df.columns}")
            
            lang_list = [hrl, lrl]
            # print(lang_list)

            for src_lang, target_lang in zip(lang_list, lang_list[-1::]) : 
                print(f"Source : {src_lang}\tTarget : {target_lang}")
            

                # K shots
                for K in [0,5,8] : 
                    print(f"K : {K}")
                    # examples
                    FEW_SHOT_EXAMPLES = [(row[src_lang], row[target_lang]) for index, row in df.head(K).iterrows()]

                    generated_translations = []
                    for src_sent in tqdm(df[f"{src_lang}"][10:], desc=f"Generating {target_lang} Translations"):
                        try :
                            prompt = template(src_sent, src_lang, target_lang, FEW_SHOT_EXAMPLES)
                            pred = pipe(
                                prompt, 
                                do_sample = True, 
                                return_full_text = False,
                                max_tokens = 4096
                            )
                            print(pred[0]["generated_text"])
                            # pred = pred["generated_content"][-1][]
                            exit()
                            generated_translations.append(pred)
                        except Exception as e :
                            print(e)
                            generated_translations.append("")
                    
                    # evaluation
                    original = df[f"{target_lang}"][10:].tolist()
                    generated = generated_translations

                    sp_bleu_score = sacrebleu.corpus_bleu(generated, [original], tokenize="spm").score
                    chrf2_score = sacrebleu.corpus_chrf(generated, [original], beta=2).score

                    print(f"{sp_bleu_score:.2f}/{chrf2_score:.2f}")

        except Exception as e : 
            print(e)
        

        
    
        