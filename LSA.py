import re
import sys
import nltk
import unidecode
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from gensim import corpora
from gensim.models import LsiModel
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt

def load_data(file_name):
    documents_list = []
    titles=[]
    with open (file_name, "r", encoding="utf8") as BigFile:
        canto = []
        for line in BigFile.readlines():
            if re.search("^\w+\s.\sCanto\s\w+$",line):
                titles.append(line.strip())
                documents_list.append(canto)
                canto.clear()
            else:
                canto.extend(line)
    #print("Total Number of Documents:",len(documents_list))
    #print("Titles are:")
    #for t in titles:
    #    print(t)
    return documents_list,titles

def preprocess_data(doc_set):
    tokenizer = RegexpTokenizer(r'\w+')
    ita_stops = set(stopwords.words('italian'))
    ita_stemmer = nltk.stem.snowball.ItalianStemmer()
    texts = list()
    for i in doc_set:
        l2str = ''.join(str(e) for e in i)
        l2str = unidecode.unidecode(l2str)
        raw = l2str.lower()
        tokens = tokenizer.tokenize(raw)
        stopped_tokens = [i for i in tokens if not i in ita_stops]
        stemmed_tokens = [ita_stemmer.stem(i) for i in stopped_tokens]
        texts.append(stemmed_tokens)
    return texts

def prepare_corpus(doc_clean):
    dictionary = corpora.Dictionary(doc_clean)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    return dictionary,doc_term_matrix

def create_gensim_lsa_model(doc_clean,number_of_topics,words):
    dictionary,doc_term_matrix=prepare_corpus(doc_clean)
    lsamodel = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary)
    print(lsamodel.print_topics(num_topics=number_of_topics, num_words=words))
    return lsamodel

def compute_coherence_values(dictionary, doc_term_matrix, doc_clean, stop, start=2, step=3):
    coherence_values = []
    model_list = []
    for number_of_topics in range(start, stop, step):
        # generate LSA model
        model = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary)  # train model
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=doc_clean, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

def plot_graph(doc_clean,start, stop, step):
    dictionary,doc_term_matrix=prepare_corpus(doc_clean)
    model_list, coherence_values = compute_coherence_values(dictionary, doc_term_matrix,doc_clean,
                                                            stop, start, step)
    # Show graph
    x = range(start, stop, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()



if __name__ == "__main__":

    texts,titles = load_data(sys.argv[1])
    clean_text = preprocess_data(texts)
    start,stop,step=2,20,1
    plot_graph(clean_text,start,stop,step)
    #number_of_topics=90
    #words=10
    #model=create_gensim_lsa_model(clean_text,number_of_topics,words)


            
        