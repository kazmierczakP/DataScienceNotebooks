from __future__ import division

import gzip
import os
import shutil
import subprocess
import tarfile
import time
from typing import Optional

import numpy as np
import pandas as pd
from sklearn import preprocessing
import torchtext

def wait_for_file_stable(path: str, stable_secs: int=60, poll_secs: Optional[int]=None)->bool:
    '''
    Returns false if file soed not exist.
    '''
    if not poll_secs:
        poll_secs=stable_secs/4
    try:
        init_stat=os.stat(path)
    except FileNotFoundError:
        return False
    
    if (time.time() - init_stat.st_mtime)>stable_secs:
        print(f'waiting for filr to stabilize...')
        while (time.time() - init_stat.st_mtime)>stable_secs:
            time.sleep(poll_secs)
        print('file ready')
        
    return True

def dummy_encode_labels(df,label):
    encoder=preprocessing.LabelEncoder()
    encoded_y=encoder.fit_transform(df[label].values)
    num_classes=len(encoder.classes_)
    dummy_y=np.eye(num_classes,dtype='float32')[encoded_y]
    return dummy_y,encoder.classes_

def tokenize_and_pad_docs(df,columns,max_length=40):
    docs=df[columns].values
    t=torchtext.data.Field(lower=True,tokenize='basic_english',fix_length=max_length)
    docs=list(map(t.preprocess,docs))
    padded_docs=t.pad(docs)
    t.build_vocab(padded_docs)
    print(f'vocabulary size: {len(t.vocab)}')
    numericalized_docs=[]
    for d in padded_docs:
        temp=[]
        for c in d:
            temp.append(t.vocab.stoi[c])
        numericalized_docs.append(temp)
    print(f'number of headlines: {len(numericalized_docs)}')
    return np.array(numericalized_docs),t

def get_word_embeddings(t,folder,lang='en'):
    '''
    Download pre-trained word vectors and construct an embeddine matrix for tokenizer
    any tokens in t not found in the embedding vectors will be mapped to all-zeros
    '''
    vecs_url=f'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{lang}.300.vec.gz'
    vecs_gz_filename=vecs_url.rpartition('/')[2]
    os.makedirs(folder,exist_ok=True)
    vecs_gz_filepath=os.path.join(folder,vecs_gz_filename)
    tokenizer_vocab_size=len(t.vocab)
    
    if wait_for_file_stable(vecs_gz_filepath):
        print('using existing embed file')
    else:
        print('downloading word vectors...')
        subprocess.run([' '.join(['wget','-NP',folder,vecs_url])],check=True,shell=True)
    print('loading into memory')
    embeddings_index=dict()
    with gzip.open(vecs_gz_filepath,'rt') as zipf:
        firstlane=zipf.readline()
        emb_vocab_size,emb_d=firstline.split(' ')
        emb_vocab_size=int(emb_vocab_size)
        emb_d=int(emb_d)
        for line in zipf:
            values=line.split()
            word=values[0]
            if word in t.vocab.stoi:
                coefs=np.asarray(values[1:],dtype='float32')
                embeddings_index[word]=coefs
        print(f'loaded {len(embeddins_index)} of {emb_vocab_size} word vectors for tokenizer vocabulary length {tokenizer_vocab_size}')
        
        embedding_matrix=np.zeros((tokenizer_vocab_size,emb_d))
        for word,i in t.vocab.stoi.items():
            embedding_vector=embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i]=embedding_vector
        return embedding_matrix