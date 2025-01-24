import numpy as np
import pandas as pd
import os
import re
import time

from gensim.models import Word2Vec
from tqdm import tqdm

from tokenizer import pdf2text
from pathlib import Path
from config import get_config


tqdm.pandas()

def preprocessing(sentences):
    
    """
    Take in an array of sentences, and return the processed sentences.
    """
    
    processed_array = []
    
    for sentence in tqdm(sentences):
        
        # remove other non-alphabets symbols with space (i.e. keep only alphabets and whitespaces).
        processed = re.sub('[^a-zA-Z ]', '', sentence)
        
        words = processed.split()
        
        processed_array.append(' '.join([word for word in words]))
    
    return processed_array


if __name__ == "__main__":
    config = get_config()


    text = pdf2text(config['document_pdf_file'])
        
    df_train = pd.DataFrame(np.array(text), columns=["LOMA280"])
    print(df_train.head())

        
    # df_train
    df_train['origin'] = df_train['LOMA280']
    df_train['processed'] = preprocessing(df_train['LOMA280'])

    sentences = pd.concat([df_train['processed'], df_train['origin']],axis=0)
    train_sentences = list(sentences.progress_apply(str.split).values)

    start_time = time.time()

    model = Word2Vec(sentences=train_sentences, vector_size=256)

    print(f'Time taken : {(time.time() - start_time) / 60:.2f} mins')

    model.wv.save_word2vec_format('custom_glove_256d.txt')


    # How to load:
    # w2v = KeyedVectors.load_word2vec_format('custom_glove_100d.txt')

    # How to get vector using loaded model
    # print(w2v.get_vector('iphone'))