import numpy as np
import pandas as pd
import random
import argparse
from nltk.corpus import wordnet 


#stop words list
STOPWORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',\
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',\
    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who',\
    'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',\
    'have', 'has', 'had', 'having', 'do', 'does', 'did','doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',\
    'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',\
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',\
    'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',\
    'where', 'why', 'how', 'all', 'any', 'both', 'each','few', 'more', 'most', 'other', 'some', 'such', 'no',\
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',\
    'should', 'now', '']


def synonym_replacement(words, n, raw_words):
    
    re_words = dict()
    new_words = raw_words.copy()
    random_word_list = words
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            re_words[random_word] = synonym
            new_words = [re_words[random_word] if word==random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n: #only replace up to n words
            break
            
    return new_words


def get_synonyms(word):
    
    synonyms = set()
    for syn in wordnet.synsets(word): 
        for l in syn.lemmas(): 
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            if " " not in synonym:
                synonyms.add(synonym) 
    if word in synonyms:
        synonyms.remove(word)
        
    return list(synonyms)


def random_deletion(p, raw_words, length):

    #randomly delete words with probability p
    new_words = []
    for word in raw_words[:length]:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)
    
    #if you end up deleting all words, just return a random word
    if len(new_words) == 0 and length != 0:
        rand_int = random.randint(0, length-1)
        new_words.append(raw_words[rand_int])
    elif length == 0:
        pass
    
    return new_words


def random_swap(words, n, length):
    
    new_words = words.copy()
    for _ in range(n):
        if length != 0:
            new_words = swap_word(new_words, length)
        else:
            pass
        
    return new_words


def swap_word(new_words, length):
    
    random_idx_1 = random.randint(0, length-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, length-1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
    
    return new_words


def random_insertion(words, n, length):
    
    new_words = words.copy()
    for _ in range(n):
        if length != 0:
            add_word(new_words, length)
        else:
            pass
        
    return new_words


def add_word(new_words, length):
    
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, length-1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, length-1)
    new_words.insert(random_idx, random_synonym)
    
    
def data_eda(x):
    
    alpha_sr = 0.1
    alpha_rs = 0.1
    alpha_ri = 0.1
    p_rd = 0.1
    num_new_per_technique = 1
    for _ in range(num_new_per_technique):
        raw_words = x
        x_len = len(x)
        words = [word for word in raw_words if word not in STOPWORDS and word[0] in 'qwertyuiopasdfghjklzxcvbnm' \
                                                and len(word) > 1]
        num_words = len(words)
        words = list(set(words))
        n_sr = max(1, int(alpha_sr*num_words))
        n_rs = max(1, int(alpha_rs*num_words))
        n_ri = max(1, int(alpha_ri*num_words))
        eda_index = random.randint(0, 3)
        if eda_index == 0:
            a_words = synonym_replacement(words, n_sr, raw_words)
        elif eda_index == 1:
            a_words = random_deletion(p_rd, raw_words, x_len)
        elif eda_index == 2:
            a_words = random_swap(raw_words, n_rs, x_len)
        elif eda_index == 3:
            a_words = random_insertion(raw_words, n_ri, x_len)
    
    return a_words


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', '--train_data', help='Name of the training data file', required=False)
    args = vars(parser.parse_args())
    
    concat_text = pd.DataFrame()
    raw_text = pd.read_csv(args['train_data'], usecols=[0], encoding='ISO-8859-1')
    raw_target = pd.read_csv(args['train_data'], usecols=[1], encoding='ISO-8859-1')
    raw_label = pd.read_csv(args['train_data'], usecols=[2], encoding='ISO-8859-1')
    seen = pd.read_csv(args['train_data'], usecols=[3], encoding='ISO-8859-1')
    concat_text = pd.concat([raw_text, raw_target, raw_label, seen], axis=1)
    
    half = len(raw_label)
    print("Length of original train set is: ", half)
    aug_text = pd.concat([concat_text]*2, ignore_index=True)
    for i in range(half, 2*half):
        aug_text['Tweet'].iloc[i] = " ".join(data_eda(aug_text['Tweet'].iloc[i].split()))
    
    print(aug_text['Tweet'].iloc[half-1])
    print(aug_text['Tweet'].iloc[-1])
    
    aug_text.to_csv("/home/yli300/EMNLP2022/data/raw_train_all_subset_eda2_kg_onecol.csv", index=False)