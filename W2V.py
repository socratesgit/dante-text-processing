import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from gensim.test.utils import common_texts
from gensim.models import Word2Vec

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
                stopped_tokens = [i for i in tokens if not i in ita_stops or len(i) == 1]
                #stemmed_tokens = [ita_stemmer.stem(i) for i in stopped_tokens]
                #bow.extend(stemmed_tokens)
                bow.extend(stopped_tokens)
            texts.append(bow)
    return texts

if __name__ == '__main__':
    with open("divina_commedia.json","r") as f_index:
        index = json.load(f_index)
    data = preprocess_data(index)
    model = Word2Vec(sentences=data, vector_size=20, window=3, min_count=2, workers=4)
    model.save("word2vec.model")    
    sims = model.wv.most_similar('amore', topn=10)
    print(sims)