import caronte as car
import PySimpleGUI as sg

sg.theme('BluePurple')

layout = [[sg.Text('Cosa vuoi cercare oggi?'), sg.Text(size=(15,1), key='-OUTPUT-')],
          [sg.Input(key='-IN-')],
          [sg.Button('Cerca'), sg.Button('Esci')]]

index = car.load_data('divina_commedia.txt')
data = car.preprocess_data(index)
titles = car.titles_json(index)
dictionary,doc_term_matrix = car.prepare_corpus(data)
model = car.create_model(doc_term_matrix)
corpus_tfidf = model[doc_term_matrix]
similarity_mtrx = car.create_sim_mtrx(corpus_tfidf) 

window = sg.Window('Caronte', layout)

while True:  # Event Loop
    event, values = window.read()
    print(event, values)
    if event == sg.WIN_CLOSED or event == 'Esci':
        break
    if event == 'Cerca':
       
        query = car.prepare_query(str(window['-IN-']),dictionary)
        vec_tdidf = model[query]
        res_query = similarity_mtrx[vec_tdidf]
        i = 0
        for doc_position, doc_score in res_query:
            if i == 1:
                break
            res_query = ''
            res_query += titles[doc_position]+'\n'
            res_query += car.find_terzina(titles[doc_position],index)+'\n'
            i += 1
        
        window['-OUTPUT-'].update(res_query)

window.close()