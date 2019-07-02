# -*- coding: utf-8 -*-

#Imports
import pandas as pd
import re
import string
import numpy as np
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import word_tokenize
from collections import Counter
import time
#from sklearn import preprocessing as spp
contractions = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there had",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "’s": "",
    "'s": "",
    "u.s.a": "america",
    "u.s": "america",
    "capt.": "captain",
    "mr.": "mister",
    "mrs.": "miss",
    "capt.": "captain"
}


def repIt(x):
    x = x.strip()
    #if all(c.isdigit() for c in x):
    #    return 'number'
    if any(c.isdigit() for c in x):
        return 'number'
    return x
wordnet = WordNetLemmatizer() 
def lemmaIt(word,tag):
    if tag.startswith("NN"):
        return wordnet.lemmatize(word,pos='n')
    elif tag.startswith("VB"):
        return wordnet.lemmatize(word,pos='v')
    elif tag.startswith("JJ"):
        return wordnet.lemmatize(word,pos='a')
    elif tag.startswith("R"):
        return wordnet.lemmatize(word,pos='r')   
    else:
        return wordnet.lemmatize(word,pos='n')  #e.g CD: numeral,IN: preposition, PR: pronoun, etc
    return word
def concatIt(lst1,lst2):
    lst1 = np.hstack((lst1,lst2))
    return
def my_pp(raw_text_df, split_sents=False, remove_fullstops=False, remove_stopwords=False):
    stime = time.time()
    #Lower case and change contractions
    raw_text = raw_text_df.copy()
    processed = raw_text.str.lower()  
    processed = pd.Series(processed)
    print("In PREPROCESS: initialization success...")
    for key in contractions:
        processed = processed.str.replace(key,contractions[key])
    print("In PREPROCESS: contarctions replaced success...")
    #reg ex replacements
    processed = processed.str.replace(r'£|\$(\ ?\d*\.?\d*)', ' currency ') 
    processed = processed.str.replace(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b',' email ')
    processed = processed.str.replace(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)',' url ')
    processed = processed.str.replace(r'(\d{4}[-.]+\d{2}[-.]+\d{2})', ' date ')
    processed = processed.str.replace(r'(\d{2}[:]+\d{2})|(\d{1}[:]+\d{2})|(\d{1}[:]+\d{2}[:]+\d{2})', ' time ')
    processed = processed.str.replace(r'(\b\d{1,2}\D{0,3})?\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|(nov|dec)(?:ember)?)\D?(\d{1,2}\D?)?\D?((19[7-9]\d|20\d{2})|\d{2})', ' date ')  
    processed = processed.str.replace(r'\b\D?((19[7-9]\d|20\d{2})|\d{2})', ' year ')
    print("In PREPROCESS: reg ex terms replaced success...")
    #Divide the descriptions into sentences
    if split_sents is True: 
        processed = fsplit_sents(processed)
        print("In PREPROCESS: divided into individual sentences success...")
    #Remove all punctuation items
    remove = string.punctuation
    if not remove_fullstops:
        remove = remove.replace(".", "")
    ptrn = r"[{}]".format(remove)
    for item in ptrn:
        processed = processed.str.replace(item,' ')
    print("In PREPROCESS: punctuations replaced success...")
    #Remove stop words
    if remove_stopwords is True:
        fremove_stopwords(processed)
        print("In PREPROCESS: stopwords removed success...")
    #Lemmatize  
    processed = processed.apply(lambda row: ' '.join([lemmaIt(word,tag) for word,tag in nltk.pos_tag(wordpunct_tokenize(row))])) 
    print("In PREPROCESS: lemmatize success...")
    #Remove all characters except which are alpha numeric
    #processed = processed.apply(lambda row: re.sub(r'[^a-zA-Z0-9=]',' ',str(row)))
    #Replace all numbers
    processed = processed.apply(lambda row: ' '.join([repIt(x) for x in wordpunct_tokenize(row)]))
    #return
    etime = time.time()
    print("PREPROCESS complete in time: ",etime-stime)
    return processed
def fsplit_sents(processed):
    lst = []
    processed = processed.apply(lambda row: lst.extend(re.split("(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s",str(row))))
    processed = pd.Series(lst)
    return processed 
def fremove_stopwords(processed):
    stop_words = stopwords.words('english') #stop_words = set(stopwords.words('english'))
    processed = processed.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words)) 
    return processed
def fremove_fullstops(processed):
    processed = processed.str.replace(r'.', ' ')
    return processed
def nUnique_tokens(processed):
    all_tokens = word_tokenize(str( ' '.join(np.array(processed)) ))
    counter = Counter(all_tokens)
    #nUnique = np.unique(all_tokens)
    #del all_tokens
    #return nUnique
    return counter

























#Stemmer
#porter = nltk.PorterStemmer()
#processed = processed.apply(lambda x: ' '.join(porter.stem(term) for term in x.split()))
#import re;removelist = "=.";mystring = "asdf1234=.!@#$";re.sub(r'[^\w'+removelist+']', '',mystring);
