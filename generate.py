from pathlib import Path
from config import get_config, latest_weights_file_path 
from GPTModel import build_GPTModel 
from tokenizers import Tokenizer
from datasets import load_dataset
from dataset import BilingualDataset
import torch
import sys

def generate_sentence(sentence: str):
    # Define the device, tokenizers, and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    config = get_config()
    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
    model = build_GPTModel(vocab_size=tokenizer_src.get_vocab_size(),
                              max_len=config["seq_len"],
                              num_head=8,
                              num_block=12,
                              d_model=512)
    
    # Load the pretrained weights
    model_filename = latest_weights_file_path(config)
    state = torch.load(model_filename, map_location=torch.device(device))
    model.load_state_dict(state['model_state_dict'])

    # translate the sentence
    model.eval()
    with torch.no_grad():
        # Print the source sentence and target start prompt
        print(f"{f'SOURCE: ':>12}{sentence}")
        print(f"{f'GENERATED: ':>12}", end='')
        
        source = tokenizer_src.encode(sentence)
        source = torch.cat([
            torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64), 
            torch.tensor(source.ids, dtype=torch.int64)
        ], dim=0).to(device)
        source = torch.reshape(source, (1, source.size(0)))
        # print(source, tokenizer_src.token_to_id('[EOS]'))
        # Generate the translation word by word
        prediction = model.generate(source, tokenizer_src.token_to_id('[EOS]'), 100)
        prediction = torch.reshape(prediction, (prediction.size(1),))
        # print(prediction)
        for i in prediction:
            print(f"{tokenizer_src.decode([i.item()])}", end=' ')
        print()

    # convert ids to tokens
    return None
    
#read sentence from argument
generate_sentence('The man ')