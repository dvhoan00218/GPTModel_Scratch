import os
from pathlib import Path
from config import get_config

import numpy as np
import pandas as pd

import pdfplumber

from datasets import Dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace



def pdf2text(path: str):
    result = []
    with pdfplumber.open(path) as temp:
        for page in temp.pages:
            paraphases = page.extract_text().split("\n")
            for paraphase in paraphases:
                result.append(paraphase)
    return result


def get_all_sentences(ds, lang):
    for item in ds:
        yield item[lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


if __name__ == "__main__":
    config = get_config()

    #Convert the PDF file into the List containing the contents of PDF file in text.
    #Example: ["This is a test. This is a test.", "mark. Hero.mark. Hero.", "Token"]
    text = pdf2text(config['document_pdf_file'])
    
    df = pd.DataFrame(np.array(text), columns=["LOMA280"])
    print(len(df))
    
    ds_raw = Dataset.from_pandas(df)
    # print(ds_raw["LOMA280"])
    
    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, "LOMA280")
