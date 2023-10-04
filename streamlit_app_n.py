import streamlit as st
from autocomplete_ngram_app import AutoCompleteModel
from autocomplete_ngram_app import Dataset  # Sesuaikan dengan modul yang sesuai
import ijson

# Buat instance model AutoComplete
model = AutoCompleteModel()

# Baca file JSON secara streaming
with open("C:\\Users\\rafli\\Documents\\Kuliah SMT 5\\NLP\\N-gram autocomplete\\Final\\model.json", 'rb') as file:
    # Gunakan ijson untuk membaca file secara streaming
    data = ijson.items(file, 'item')
    
    # Loop melalui setiap item dalam file JSON
    for item in data:
        # Proses item sesuai dengan kebutuhan Anda
        model.load_from_json(item)

# Buat instance objek Dataset
ds = Dataset()

# Kemudian lakukan penggunaan objek ds sesuai kebutuhan Anda

# Tokenize data sebelum melatih model
model.tokenize(ds.lines)

# Latih model dengan mengatur ngram saat train
model.train(minimum_freq=1, ngram=3)  # Atur _ngram ke nilai yang sesuai (misalnya, 3)

st.title('Autocomplete Text Prediction')

# Tambahkan input teks sebelumnya
previous_text = st.text_input('Enter Previous Text:', '')

if previous_text:
    suggestions = model.suggestions(previous_text.split(), num_suggestions=5)
    st.write('Predictions:')
    for suggestion in suggestions:
        st.write(suggestion)
