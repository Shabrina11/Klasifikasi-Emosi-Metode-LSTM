from flask import Flask, render_template, request
import numpy as np
from function import globalProcess
from sklearn.metrics import confusion_matrix, accuracy_score
from werkzeug.utils import secure_filename
import pandas as pd
import ast
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.utils import to_categorical
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/', methods=["POST", "GET"])
def home():
    prediction = "-"
    if request.method=="POST":
        InputUser = request.form["pred"]
        preprocess = globalProcess().clean_text(InputUser)
        ttsr = globalProcess().tts(preprocess)
        lmr = globalProcess().predict(ttsr)
        return render_template('index.html', prediction=lmr)

    else: return render_template('index.html', prediction=prediction) 

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Menerima file yang diunggah dari formulir
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            file_path = 'Deployment/' + secure_filename(uploaded_file.filename)
            uploaded_file.save(file_path)
            data = pd.read_csv(file_path)

            processor = globalProcess()
            
            # text preprocessing
            data['Tweet_clean'] = data['Tweet'].apply(lambda x: processor.clean_text(str(x)))
            data.to_csv('Data bersih.csv', index=False)
            df = pd.read_csv('Data bersih.csv')
            
            labels = df['Label']
            tweets = df['Tweet_clean']
            
            # Pembagian data latih dan data uji
            from sklearn.model_selection import train_test_split

            X = globalProcess().ttsequence(tweets)
            y = labels

            # global X_train, X_test, y_train, y_test
            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, stratify=y, random_state=42)
            
            # One-hot encoding 
            y_train = to_categorical(y_train, 5)
            y_test = to_categorical(y_test, 5)
            
            # Penggunaan model LSTM
            model = load_model("C:/Kode LSTM/LSTM Emotion Classification/Deployment/TestModelFix.h5")
            y_pred = model.predict(X_test)
            y_pred = np.argmax(y_pred, axis=1)
            y_test = np.argmax(y_test, axis=1)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Kinerja
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')  
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            return render_template('result.html', confusion_matrix=cm, accuracy=accuracy, precision=precision, recall=recall, f1=f1)
        
if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=8080)
