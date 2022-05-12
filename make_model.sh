#!/bin/bash

echo "use bandcamp; SELECT $1 FROM albums;" |
mysql -u coen --password=AcidBearCanoe#3 -h 192.168.1.18 > data/"$1"_disorganized.txt

tr -d '[]",' < data/"$1"_disorganized.txt > data/"$1"_sentences.txt
rm data/"$1"_disorganized.txt

./make_word2vec.py "$1" < data/"$1"_sentences.txt