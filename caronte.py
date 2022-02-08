import re
import json
import unidecode
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from gensim import models
from gensim.corpora import Dictionary
from gensim import similarities

def load_data(file_name):
    index = {}
    with open (file_name, "r", encoding="utf8") as BigFile:
        titolo_canto = ''
        num_terzina = 0
        verso_terzina = 0
        for line in BigFile.readlines():
            line = unidecode.unidecode(line)
            if re.search("^\w+\s.\sCanto\s\w+$",line):
                line = line.strip()
                index[line] = {}
                titolo_canto = line
                num_terzina = 0
                verso_terzina = 0
            elif line == '\n':
                continue
            else:
                k_terzina = str(num_terzina)
                k_verso = str(verso_terzina)
                if verso_terzina == 0:
                    index[titolo_canto][k_terzina] = {}
                index[titolo_canto][k_terzina][k_verso] = line.strip('\n')
                if verso_terzina == 2:
                    verso_terzina = 0
                    num_terzina += 1
                else:
                    verso_terzina += 1

    a_file = open("./divina_commedia.json", "w")
    json.dump(index, a_file)
    a_file.close()

    #print("Total Number of Documents:",len(documents_list))
    #print("Titles are:")
    #for t in titles:
    #    print(t)
    return index

def titles_json(doc_set):

    titles = list()
    for canto in doc_set:
        for terzina in doc_set[canto]:
            t = canto+'_'+terzina
            titles.append(t)
    return titles

def preprocess_data(doc_set):

    tokenizer = RegexpTokenizer(r'\w+')
    ita_stops = set(stopwords.words('italian'))
    ita_stemmer = nltk.stem.snowball.ItalianStemmer()
    texts = list()
    for canto in doc_set:
        for terzina in doc_set[canto]:
            bow = list()
            for verso in doc_set[canto][terzina]:
                raw = doc_set[canto][terzina][verso].lower()
                tokens = tokenizer.tokenize(raw)
                stopped_tokens = [i for i in tokens if not i in ita_stops]
                stemmed_tokens = [ita_stemmer.stem(i) for i in stopped_tokens]
                bow.extend(stemmed_tokens)
            texts.append(bow)
    return texts

def prepare_corpus(doc_clean):
    dictionary = Dictionary(doc_clean)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    return dictionary,doc_term_matrix

def prepare_query(query,dictionary):
    
    tokenizer = RegexpTokenizer(r'\w+')
    ita_stops = set(stopwords.words('italian'))
    ita_stemmer = nltk.stem.snowball.ItalianStemmer()
    query = unidecode.unidecode(query)
    raw = query.lower()
    tokens = tokenizer.tokenize(raw)
    stopped_tokens = [i for i in tokens if not i in ita_stops]
    final = [ita_stemmer.stem(i) for i in stopped_tokens]
    vec_bow = dictionary.doc2bow(final)
    return vec_bow

def find_terzina(title,index):
    canto, terzina = tuple(title.split("_"))
    output = ''
    for key in index[canto][terzina]:
        output += index[canto][terzina][key]+'\n'
    return output

def create_model(doc_term_matrix):
    return TfidfModel(doc_term_matrix)

def create_sim_mtrx(corpus_tfidf):
    return similarities.MatrixSimilarity(corpus_tfidf) 

if __name__ == '__main__':
    try:
        with open("divina_commedia.json","r") as f_index:
            index = json.load(f_index)
    except IOError:
        index = load_data('divina_commedia.txt')
    data = preprocess_data(index)
    titles = titles_json(index)
    dictionary,doc_term_matrix = prepare_corpus(data)
    model = models.TfidfModel(doc_term_matrix)
    corpus_tfidf = model[doc_term_matrix]
    similarity_mtrx = similarities.MatrixSimilarity(corpus_tfidf) 
    while True:
        query = input("What do you want to look for today?\n")
        if(not query):
            break
        print("\nYour query:")
        print(query+'\n')
        query = prepare_query(query,dictionary)
        vec_tdidf = model[query]
        res_query = similarity_mtrx[vec_tdidf]
        res_query = sorted(enumerate(res_query), key = lambda item: -item[1])
        i = 0
        for doc_position, doc_score in res_query:
            if i < 5:
                print(doc_score)
                print(titles[doc_position])
                print(find_terzina(titles[doc_position],index))
                i += 1

    
    

    
    