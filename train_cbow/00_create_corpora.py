import collections
import pickle
import re

import requests
from datasets import get_dataset_split_names, load_dataset
from cbow_utils import preprocess  # Assuming preprocess is defined in cbow_utils.py

def concat_and_process(strings, delimiter="<DELIMIT>"):
    concatenated = f" {delimiter} ".join(strings)
    return preprocess(concatenated, min_count=5)

# Fetch text8 dataset as a starting point.
r = requests.get('https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8')
with open('text8', 'wb') as f: f.write(r.content)
with open('text8') as f: text8: str = f.read() # i.e. load the text8 corpus into a variable (named text8)
text8 = text8.replace("<", "").replace(">", "")

# Also collect queries and answers from MS Marco dataset?
dataset = load_dataset("microsoft/ms_marco", "v1.1", cache_dir="./data/marco")
train_data = dataset["train"]
marco_corpus = []
for sample in train_data:
    query = sample.get("query", "")
    answers = sample.get("answers", [])
    full_text = (query + " " + " ".join(answers)).replace("<", "").replace(">", "")
    marco_corpus.append(full_text.strip())

print(f"text8 type: {type(text8)}")
print(f"marco_corpus type: {type(marco_corpus)}")
print(f"len(marco_corpus): {len(marco_corpus)}")

corpus: list[str] = concat_and_process([text8] + marco_corpus)
corpus = ['<DELIMIT>' if word == '<delimit>' else word for word in corpus]
print(type(corpus)) # <class 'list'>
print(len(corpus))  # 
print(corpus[:7])   # ['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse']
print(corpus[-7:])

# once saved, check content with: head -c 100 corpus.json
with open('corpus.pkl', 'wb') as f: pickle.dump(corpus, f)

#
#
#
def create_lookup_tables(words: list[str]) -> tuple[dict[str, int], dict[int, str]]:
  word_counts = collections.Counter(words)
  vocab = sorted(word_counts, key=lambda k: word_counts.get(k), reverse=True)
  int_to_vocab = {ii+1: word for ii, word in enumerate(vocab)}
  int_to_vocab[0] = '<UNK>'
  vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
  return vocab_to_int, int_to_vocab


#
#
#
words_to_ids, ids_to_words = create_lookup_tables(corpus)
tokens = [words_to_ids[word] for word in corpus]
print(type(tokens)) # <class 'list'>
print(len(tokens))  # 
print(tokens[:7])   # 
print(tokens[-7:])  #


#
#
#
print(ids_to_words[5724])        # 'anarchism
print(words_to_ids['anarchism']) # 5724
print(len(words_to_ids))         # 69061
print(f"<DELIMIT> ID: {words_to_ids['<DELIMIT>']}") # 30


#
#
#
with open('tkn_words_to_ids.pkl', 'wb') as f: pickle.dump(words_to_ids, f)
with open('tkn_ids_to_words.pkl', 'wb') as f: pickle.dump(ids_to_words, f)