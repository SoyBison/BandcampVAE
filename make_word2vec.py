#!/usr/bin/env python

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import argparse
import io
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str)
    args = parser.parse_args()

    text = sys.stdin.read()
    text_flo = io.StringIO(text)
    sentences = LineSentence(text_flo)
    model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=8)
    model.save(f'models/{args.name}.model')
