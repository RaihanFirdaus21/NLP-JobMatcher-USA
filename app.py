import streamlit as st
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load dataset
final_data = pd.read_csv('dataset.csv')

final_data = final_data[final_data.state != 'No Info']
final_data['state'] = final_data['state'].str.title()
final_data['city'] = final_data['city'].str.title()
final_data['tipe_pendaftaran'] = final_data['tipe_pendaftaran'].replace({
    'ComplexOnsiteApply': 'Complex Onsite Apply', 
    'OffsiteApply': 'Offsite Apply', 
    'SimpleOnsiteApply': 'Simple Onsite Apply'
})

# Load precomputed embeddings and dataset if available (to save time)
with open('model_components.pkl', 'rb') as f:
    components = pickle.load(f)

# Load GloVe vectors
glove_vectors = components['glove_vectors']
corpus_embeddings = components['corpus_embeddings']

# Preprocessing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
num_features = 300

def get_average_glove(tokens, model, num_features):
    valid_words = [word for word in tokens if word in model]
    if not valid_words:
        return np.zeros(num_features)
    return np.mean([model[word] for word in valid_words], axis=0)

# Initialize session state variables
if 'selected_state' not in st.session_state:
    st.session_state.selected_state = 'Any'
if 'selected_city' not in st.session_state:
    st.session_state.selected_city = 'Any'
if 'selected_type' not in st.session_state:
    st.session_state.selected_type = 'Any'
if 'selected_sponsor' not in st.session_state:
    st.session_state.selected_sponsor = 'Any'
if 'selected_apply' not in st.session_state:
    st.session_state.selected_apply = 'Any'

# Streamlit app
st.title('Job Recommendation System')

# Dropdown for category selection
state_option = final_data['state'].unique()
state_option = np.sort(state_option)
type_options = final_data['jenis_pekerjaan'].unique()
sponsor_options = final_data['disponsori'].unique()
apply_options = final_data['tipe_pendaftaran'].unique()

state_option = np.concatenate([np.array(['Any']), state_option])
type_options = np.concatenate([np.array(['Any']), type_options])
sponsor_options = np.concatenate([np.array(['Any']), sponsor_options])
apply_options = np.concatenate([np.array(['Any']), apply_options])

selected_state = st.selectbox('Select State', state_option, index=np.where(state_option == st.session_state.selected_state)[0][0])
st.session_state.selected_state = selected_state

if selected_state == 'Any':
    city_option = ['Any']
else:
    city_option = final_data[final_data['state'] == selected_state]['city'].unique()
    city_option = np.sort(city_option)
    city_option = np.insert(city_option, 0, 'Any')

selected_city = st.selectbox('Select City', city_option, index=np.where(city_option == st.session_state.selected_city)[0][0])
st.session_state.selected_city = selected_city

selected_type = st.selectbox('Select Job Type', type_options, index=np.where(type_options == st.session_state.selected_type)[0][0])
st.session_state.selected_type = selected_type

selected_sponsor = st.selectbox('Select Sponsor Type', sponsor_options, index=np.where(sponsor_options == st.session_state.selected_sponsor)[0][0])
st.session_state.selected_sponsor = selected_sponsor

selected_apply = st.selectbox('Select Application Type', apply_options, index=np.where(apply_options == st.session_state.selected_apply)[0][0])
st.session_state.selected_apply = selected_apply

# Filter data based on selected categories
city_filter = (final_data['city'] == selected_city) if selected_state != 'Any' and selected_city != 'Any' else (final_data['state'] == selected_state if selected_state != 'Any' else final_data['city'] == final_data['city'])
type_filter = (final_data['jenis_pekerjaan'] == selected_type) if selected_type != 'Any' else (final_data['jenis_pekerjaan'] == final_data['jenis_pekerjaan'])
sponsor_filter = (final_data['disponsori'] == selected_sponsor) if selected_sponsor != 'Any' else (final_data['disponsori'] == final_data['disponsori'])
apply_filter = (final_data['tipe_pendaftaran'] == selected_apply) if selected_apply != 'Any' else (final_data['tipe_pendaftaran'] == final_data['tipe_pendaftaran'])

# Text box to input new skill description
new_text = st.text_area('Enter New Skill Description')

if st.button('Search Jobs'):
    if new_text:
        filtered_data = final_data[(city_filter) & (type_filter) & (sponsor_filter) & (apply_filter)]
        
        if filtered_data.shape[0] == 0:
            st.write('Too Many Filters, Reduce Filters!')
        else:
            filtered_indices = filtered_data.index
            filtered_embeddings = corpus_embeddings[filtered_indices]

            input_tokens = word_tokenize(new_text.lower())
            input_tokens = [lemmatizer.lemmatize(word) for word in input_tokens if word not in stop_words]
            input_embedding = get_average_glove(input_tokens, glove_vectors, num_features)

            cosine_sim_new = cosine_similarity([input_embedding], filtered_embeddings).flatten()
            top_3_indices = cosine_sim_new.argsort()[-3:][::-1]
            top_3_jobs = filtered_data.iloc[top_3_indices][['judul', 'url_posting_pekerjaan']]

            isBreak = False
            for i, index in enumerate(top_3_indices, 1):
                if cosine_sim_new[index] < 0.6:
                    st.write('Your Skill Description is Too Short, Add More Details!')
                    isBreak = True
                    break
            
            if not isBreak:
                st.write("Top 3 Jobs for You:")
                for i, row in enumerate(top_3_jobs.itertuples(), 1):
                    title_capitalized = row.judul.title()
                    st.markdown(f"{i}. **{title_capitalized}**<br>{row.url_posting_pekerjaan}", unsafe_allow_html=True)
            
    else:
        st.write("Enter a skill description to search for matching jobs.")
