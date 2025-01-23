import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load semua model dan vectorizer
# Model
model_svm = joblib.load('svm_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_svm_model.pkl')

model_svm_smote = joblib.load('svm_smote_model.pkl')
tfidf_vectorizer_smote = joblib.load('tfidf_svm_smote_model.pkl')

# Model baru dengan seleksi fitur Chi-Square
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
    # Transformasi teks menggunakan TF-IDF
    text_tfidf = vectorizer.transform([text]).toarray()
    
    # Jika menggunakan seleksi fitur, pilih hanya fitur terpilih
    if selected_features is not None:
        text_tfidf_df = pd.DataFrame(text_tfidf, columns=vectorizer.get_feature_names_out())
        text_tfidf = text_tfidf_df[selected_features].to_numpy()
    
    # Prediksi menggunakan model
    prediction = model.predict(text_tfidf)
    
    # Interpretasi hasil
    sentiment = "Positif" if prediction == 1 else "Negatif"
    return sentiment

# Header halaman
st.title('Analisis Sentimen Program Makan Bergizi Gratis dengan SVM dan Chi-Square')
st.write("""
Aplikasi ini memungkinkan Anda untuk memprediksi sentimen dari sebuah teks atau file menggunakan berbagai model Support Vector Machine.
""")

# Pilihan model untuk pengguna
model_choice = st.selectbox(
    "Pilih Model untuk Analisis Sentimen",
    (
        "SVM Standar", 
        "SVM + SMOTE", 
        "Chi-Square 80% Fitur", 
        "Chi-Square 75% Fitur", 
        "Chi-Square 50% Fitur"
    )
)

# Pilihan input: Teks manual atau file
input_choice = st.radio("Pilih jenis input:", ("Input Teks Manual", "Unggah File"))

if input_choice == "Input Teks Manual":
    # Input teks dari pengguna
    text_input = st.text_area('Masukkan teks untuk analisis sentimen')

    # Tombol untuk memprediksi sentimen
    if st.button('Prediksi Sentimen'):
        if text_input:
            # Tentukan model, vectorizer, dan fitur berdasarkan pilihan pengguna
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

            # Prediksi sentimen
            sentiment = predict_sentiment(text_input, selected_model, selected_vectorizer, selected_features)
            st.success(f'Sentimen Prediksi ({model_choice}): {sentiment}')
        else:
            st.warning('Silakan masukkan teks terlebih dahulu')

elif input_choice == "Unggah File":
    uploaded_file = st.file_uploader("Unggah file CSV atau Excel", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Membaca file
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        # Memastikan ada kolom teks
        if "teks" in data.columns:
            st.write("Data berhasil dimuat. Contoh data:")
            st.write(data.head())

            if st.button('Prediksi Semua Data'):
                # Tentukan model, vectorizer, dan fitur berdasarkan pilihan pengguna
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

                # Prediksi untuk semua data
                data["Sentimen"] = data["teks"].apply(lambda x: predict_sentiment(x, selected_model, selected_vectorizer, selected_features))

                st.write("Hasil Prediksi:")
                st.write(data)

                # Unduh hasil prediksi
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button("Unduh Hasil Prediksi", data=csv, file_name="hasil_prediksi.csv", mime="text/csv")
        else:
            st.error("File harus memiliki kolom bernama 'teks' untuk analisis sentimen.")

# Footer
st.markdown("""
---
Dibuat dengan ❤ menggunakan Streamlit
""")