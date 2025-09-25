def resize_embeddings(module, new_vocab_size):
    try : 
        old_emb = module.embed_tokens
        new_emb = nn.Embedding(new_vocab_size, old_emb.embedding_dim)
        new_emb.weight.data[:old_emb.num_embeddings] = old_emb.weight.data
        module.embed_tokens = new_emb
        return True
    
    except Exception as e : 
        print(e)
        return False




def save_special_tokens(special_tokens, tokenizer, model, path = "indic_indic_dist_320M_mod") :
    try : 
        old_tokenizer_len = len(tokenizer)

        tokenizer.add_special_tokens(
            {"additional_special_tokens":special_tokens}
        )

        model.config.encoder_vocab_size = len(tokenizer)
        model.config.decoder_vocab_size += len(tokenizer) - old_tokenizer_len
        model.config.vocab_size = model.config.decoder_vocab_size
        
        if not resize_embeddings(model.model.encoder, model.config.encoder_vocab_size) :
            raise Exception(f"cant modify encoder's embedding")

        if not resize_embeddings(model.model.decoder, model.config.decoder_vocab_size) : 
            raise Exception("cant modify decoder's embedding")

        model.lm_head.weight = model.model.decoder.embed_tokens.weight

        tokenizer.save_pretrained(path)
        model.save_pretrained(path)
        return True

    except Exception as e: 
        print(e)
        return False



def modify_token (inputs, tokenizer, token : str, if_tgt : bool = True) :
    try :
        pad_token_id = tokenizer.pad_token_id

        try : 
            token_id = tokenizer.convert_tokens_to_ids(token)
        except : 
            raise Exception(f"{token} not found in dictionary")

        # check in each input seq
        for i in range(inputs.input_ids.shape[0]): 
            for j in range(inputs.input_ids.shape[1]):

                if inputs.input_ids[i][j] != pad_token_id : 
                    # for tgt token
                    if if_tgt :
                        inputs.input_ids[i][j+1] = token_id
                    # for src token
                    else : 
                        inputs.input_ids[i][j] = token_id
                    break

    except Exception as e :
        print(e)


def pr(variable) :
    from varname import nameof
    print(f"{nameof(variable)} : {variable}")