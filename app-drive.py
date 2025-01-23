import streamlit as st 
import joblib
import numpy as np
import pandas as pd
import gdown
import os

# Fungsi untuk mendownload file dari Google Drive
def download_file_from_gdrive(url, output):
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

# URL file di Google Drive
file_urls = {
    "svm_smote_model.pkl": "https://drive.google.com/file/d/1gMILqfyNWSJiiHLMavxHNS8AISeqilAv/view?usp=sharing",
    "tfidf_svm_smote_model.pkl": "https://drive.google.com/file/d/16ZENnR5oIIf0gJBhbmETh6s46K5-OTrL/view?usp=sharing",
    
    "svm_model.pkl": "https://drive.google.com/file/d/1dSwH0I37Y1FKLJ469V4yIY-xTJPmg4Ku/view?usp=sharing",
    "tfidf_svm_model.pkl": "https://drive.google.com/file/d/1dSwH0I37Y1FKLJ469V4yIY-xTJPmg4Ku/view?usp=sharing",
    
    "svm_smote_chi-square_80_model.pkl": "https://drive.google.com/file/d/1COj1mqbwP_yv6li2HchEPzpcuvbxZDB3/view?usp=sharing",
    "tfidf_svm_smote_chi-square_80_model.pkl": "https://drive.google.com/file/d/1rsjKxPlN8cIsk_7kPn8Y8zN5MZK4FomR/view?usp=sharing",
    "selected_features_80.pkl": "https://drive.google.com/file/d/1yaBUHSDSW-_iGrztP366z0oTsrb8VQcC/view?usp=sharing",
    
    "svm_smote_chi-square_75_model.pkl": "https://drive.google.com/file/d/1QpzHOTZiWI5DpSQpfXShLMPuJ5RAIXId/view?usp=sharing",
    "tfidf_svm_smote_chi-square_75_model.pkl": "https://drive.google.com/file/d/1jWKNGw5RSgLu3pNjkiHl5WOtU5YdDqOg/view?usp=sharing",
    "selected_features_75.pkl": "https://drive.google.com/file/d/1ZMGZqGizY3plEcjIPhZ8fUewwqBPiv3q/view?usp=sharing",
    
    "svm_smote_chi-square_50_model.pkl": "https://drive.google.com/file/d/1uungmlrkMIoXxeDw3KULwXlSDHbaZeLS/view?usp=sharing",
    "tfidf_svm_smote_chi-square_50_model.pkl": "https://drive.google.com/file/d/1p9O7IjQtaKcienUB4bXHX0bbaunb_C9M/view?usp=sharing",
    "selected_features_50.pkl": "https://drive.google.com/file/d/1n_ptXhYPOSf-ehq3ABN8eMnkFjXa2-uZ/view?usp=sharing"
}

# Download semua file jika belum ada
for file_name, url in file_urls.items():
    download_file_from_gdrive(url, file_name)

# Load semua model dan vectorizer
model_svm_smote = joblib.load('svm_smote_model.pkl')
tfidf_vectorizer_smote = joblib.load('tfidf_svm_smote_model.pkl')

model_svm = joblib.load('svm_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_svm_model.pkl')

model_chi_80 = joblib.load('svm_smote_chi-square_80_model.pkl')
tfidf_chi_80 = joblib.load('tfidf_svm_smote_chi-square_80_model.pkl')
selected_features_80 = joblib.load('selected_features_80.pkl')

model_chi_75 = joblib.load('svm_smote_chi-square_75_model.pkl')
tfidf_chi_75 = joblib.load('tfidf_svm_smote_chi-square_75_model.pkl')
selected_features_75 = joblib.load('selected_features_75.pkl')

model_chi_50 = joblib.load('svm_smote_chi-square_50_model.pkl')
tfidf_chi_50 = joblib.load('tfidf_svm_smote_chi-square_50_model.pkl')
selected_features_50 = joblib.load('selected_features_50.pkl')

# Fungsi untuk memprediksi sentimen
def predict_sentiment(text, model, vectorizer, selected_features=None):
    text_tfidf = vectorizer.transform([text]).toarray()
    if selected_features is not None:
        text_tfidf_df = pd.DataFrame(text_tfidf, columns=vectorizer.get_feature_names_out())
        text_tfidf = text_tfidf_df[selected_features].to_numpy()
    prediction = model.predict(text_tfidf)
    return "Positif" if prediction == 1 else "Negatif"

# Header halaman
st.title('Analisis Sentimen dengan Model SVM')
st.write("Aplikasi ini memungkinkan Anda untuk memprediksi sentimen dari sebuah teks menggunakan berbagai model Support Vector Machine.")

# Pilihan model untuk pengguna
model_choice = st.selectbox(
    "Pilih Model untuk Analisis Sentimen",
    (
        "SVM + SMOTE", 
        "SVM Standar", 
        "Chi-Square 80% Fitur", 
        "Chi-Square 75% Fitur", 
        "Chi-Square 50% Fitur"
    )
)

# Input teks dari pengguna
text_input = st.text_area('Masukkan teks untuk analisis sentimen')

# Tombol untuk memprediksi sentimen
if st.button('Prediksi Sentimen'):
    if text_input:
        if model_choice == "SVM + SMOTE":
            selected_model = model_svm_smote
            selected_vectorizer = tfidf_vectorizer_smote
            selected_features = None
        elif model_choice == "SVM Standar":
            selected_model = model_svm
            selected_vectorizer = tfidf_vectorizer
            selected_features = None
        elif model_choice == "Chi-Square 80% Fitur":
            selected_model = model_chi_80
            selected_vectorizer = tfidf_chi_80
            selected_features = selected_features_80
        elif model_choice == "Chi-Square 75% Fitur":
            selected_model = model_chi_75
            selected_vectorizer = tfidf_chi_75
            selected_features = selected_features_75
        elif model_choice == "Chi-Square 50% Fitur":
            selected_model = model_chi_50
            selected_vectorizer = tfidf_chi_50
            selected_features = selected_features_50
        
        sentiment = predict_sentiment(text_input, selected_model, selected_vectorizer, selected_features)
        st.success(f'Sentimen Prediksi ({model_choice}): {sentiment}')
    else:
        st.warning('Silakan masukkan teks terlebih dahulu')

# Footer
st.markdown("---\nDibuat dengan ❤️ menggunakan Streamlit")
