import sys
import string
import json
import unidecode
import nltk
import numpy as np
import pandas as pd
import networkx as nx
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
#import PyPDF2

def merge_dict(dest, aux):
    for k_aux in aux:
        if k_aux in dest.keys():
            dest[k_aux]['count'] += aux[k_aux]['count']
            dest[k_aux]['position'].extend(aux[k_aux]['position'])
        else:
            dest[k_aux] = aux[k_aux]

def distance_matrix(index,k):
    mtrx = list()
    for i in index:
        arr = list()
        for j in index:
            if i == j:
                arr.append(0)
                continue
            occ = 0
            for pos_i in index[i]['position']:
                for pos_j in index[j]['position']:
                    if abs(pos_i - pos_j) <= k:
                        occ += 1
            arr.append(occ/(len(index[i]['position']) + len(index[j]['position'])))
        mtrx.append(arr)
    return mtrx
     

if __name__ == "__main__":

    #pdfName = 'path\Tutorialspoint.pdf'
    #read_pdf = PyPDF2.PdfFileReader(pdfName)

    #for i in xrange(read_pdf.getNumPages()):
    #    page = read_pdf.getPage(i)
    #    data = page.extractText()

    ita_stops = set(stopwords.words('italian'))
    tokenizer = RegexpTokenizer(r'\w+')
    ita_stemmer = nltk.stem.snowball.ItalianStemmer()
    index = dict()
    meta = dict()

    with open (sys.argv[1], "r", encoding="utf8") as BigFile:
        data=BigFile.readlines()
        num_word = 0
        for i in range(len(data)):
            #formatting in unicode
            line = unidecode.unidecode(data[i])
            #lowercase e tokenization 
            nltk_tokens = tokenizer.tokenize(line.lower())
            result = dict()
            ordered_tokens = set()
            for word in nltk_tokens:
                #stemming
                word = ita_stemmer.stem(word)
                #stopword
                if word not in ita_stops:
                    if word not in ordered_tokens:
                        ordered_tokens.add(word)
                        result[word] = {
                            "count" : 1,
                            "position" : [num_word]
                        }
                    else:
                        result[word]['count'] += 1
                        result[word]['position'].append(num_word)
                num_word += 1

            merge_dict(index,result)

        meta = {
            'num_word_tot' : num_word,
            'num_dist_word' : len(index),
            'num_line' : len(data)
        }
    
    a_file = open("./index.json", "w")
    json.dump(index, a_file)
    a_file.close()

    b_file = open("./meta.json", "w")
    json.dump(meta, b_file)
    b_file.close()

    

    
        


