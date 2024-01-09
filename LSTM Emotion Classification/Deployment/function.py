import pandas as pd
import re
import numpy as np
import ast
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

class globalProcess(object):
    #Case Folding & Noise Removal
    def clean_text(self, text):
        text = text.lower() # lower case / case folding
        text = re.sub(r"https\S+|www\S+https\S+", '',text, flags=re.MULTILINE) # hapus url
        text = re.sub(r'#[A-Za-z0-9]+','',text) # hapus hastag
        text = re.sub(r'@[A-Za-z0-9]+','',text) # hapus mention @
        text = re.sub("'", "", text) # hapus kutip
        text = re.sub('[()!?]', ' ', text) # hapus tanda baca
        text = re.sub('\[.*?\]',' ', text) # hapus kurung siku
        text = re.sub("[^a-z0-9]"," ", text) # hapus karakter
        text = re.sub(r'\d+', '', text) # hapus angka
    
    #Normalisasi
        dict_koreksi = {}
        file = open("c:/Kode LSTM/LSTM Emotion Classification/Deployment/list_norm.txt")
        for x in file:
            f = x.split(":") #memisahkan :
            dict_koreksi.update({f[0].strip(): f[1].strip()})

        for awal, pengganti in dict_koreksi.items():
            text = re.sub(r"\b" + awal + r"\b", pengganti, text)
    
    # Filtering
        # Stopword Sastrawi
        factory = StopWordRemoverFactory()
        stopword_sastrawi = factory.get_stop_words()

        text = text.split() # pisah jadi kata per kata
        text = [w for w in text if w not in stopword_sastrawi] # hapus stopwords
        text = " ".join(w for w in text) # gabung kata jadi teks

        # Stopword NLTK
        from nltk.corpus import stopwords

        stopword_nltk = set(stopwords.words("indonesian")) # set stopwords indonesia
        stopword_nltk = stopword_nltk

        text = text.split() # pisah jadi kata per kata
        text = [w for w in text if w not in stopword_nltk] # hapus stopwords
        text = " ".join(w for w in text) # gabung kata jadi teks

        # Stopword tambahan
        with open("c:/Kode LSTM/LSTM Emotion Classification/Deployment/list_stopword_tambahan.txt", "r") as f:
            stopwords_tambahan = f.read().splitlines()

        text = text.split() # pisah jadi kata per kata
        text = [w for w in text if w not in stopwords_tambahan] # hapus stopwords
        text = " ".join(w for w in text) # gabung kata jadi teks
    
    # Stemming
        # Stemming Sastrawi
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        factory = StemmerFactory()
        stemmer_sastrawi = factory.create_stemmer()

        text = stemmer_sastrawi.stem(text)
    
    # Tokenisasi
        text = nltk.word_tokenize(text)

        return text

    # Text to sequence
    def tts(self, text):
        dataset = pd.read_csv("C:/Kode LSTM/LSTM Emotion Classification/Deployment/Data/Hasil Proses Fix.csv")
        dataset["Tweet_clean"] = dataset["Tweet_clean"].apply(lambda x: ast.literal_eval(x)) # representasi list python
        X = dataset["Tweet_clean"].tolist()
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X)
        sequences = tokenizer.texts_to_sequences(X)
        max_length = max([len(s) for s in sequences])
        sequence = tokenizer.texts_to_sequences([text])
        padding = pad_sequences(sequence, maxlen=max_length)
        return padding

    # Penggunaan Model
    def predict(self, content):
        model = load_model("C:/Kode LSTM/LSTM Emotion Classification/Deployment/TestModelFix.h5")
        prediction = model.predict(content)
        # label = ["0", "1", "2", "3", "4"]
        result = np.argmax(prediction) # mengembalikan index dgn nilai terbesar
        return result
    
    def ttsequence(self, text):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(text)
        sequences = tokenizer.texts_to_sequences(text)
        max_length = max([len(s) for s in sequences])
        text = pad_sequences(sequences, maxlen=max_length)
        return text
    

     
     
    
