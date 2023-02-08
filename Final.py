
import nltk
nltk.download('wordnet')
import pandas as pd
import warnings
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import stopwords, wordnet
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from rake_nltk import Rake
import pickle
import streamlit as st
import numpy as np
from nltk.stem import PorterStemmer,WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS

pickle_out = open("lg_model.pkl","rb")
hotel=pickle.load(pickle_out)

pickle_out = open("tfidf.pkl","rb")
hotel1=pickle.load(pickle_out)

#Background image
import base64
def add_bg_from_local(cry):
    with open(cry, "rb") as cry:
        encoded_string = base64.b64encode(cry.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: 1024x768
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('new.jpg') 

# <h2 style="color:white;text-align:center">
def main():
    
    def welcome(w):
        st.markdown(f'<p style="background-color:#f4c2c2 ;color:#4b5320;font-size:24px;border-radius:2%;text-align:center">{w}</p>', unsafe_allow_html=True)
    welcome("WELCOME ALL")

    def stream(s):
        st.markdown(f'<p style="background-color:tomato;padding:10px ;color:black;font-size:24px;border-radius:2%;text-align:center">{s}</p>', unsafe_allow_html=True)
    stream("STREAMLIT HOTEL RATING NLP APP")
    
    # giving a title
    def title(t):
        st.markdown(f'<p style="background-color:#f4c2c2;padding:10px ;color:#4b5320;font-size:24px;border-radius:2%;text-align:center;font-size:28px">{t}</p>', unsafe_allow_html=True)
    title("YOUR VALUABLE REVIEWS PLEASE")
if __name__ == '__main__':
    main()

def head(url):
        st.markdown(f'<p style="background-color:tomato ;color:black;font-size:24px;border-radius:2%;text-align:center">{url}</p>', unsafe_allow_html=True)

head("(人◕‿◕) 𝔼𝕟𝕥𝕖𝕣 text 𝕥𝕠 𝕔𝕙𝕖𝕔𝕜 (人◕‿◕)")
    
# st.header("Predict Ratings for Hotel Reviews")
# st.subheader("Enter the review to analyze")

input_text = st.text_area("TYPE YOUR FEEDBACK HERE", height=80)
    
    
if st.button("Predict Rating"):
      
       wordnet=WordNetLemmatizer()
       text=re.sub('[^A-za-z0-9]',' ',input_text)
       text=text.lower()
       text=text.split(' ')
       text = [wordnet.lemmatize(word) for word in text if word not in (stopwords.words('english'))]
       text = ' '.join(text)
       pickle_out = open(r"lg_model.pkl", 'rb') 
       hotel= pickle.load(pickle_out)
       pickle_out = open(r"tfidf.pkl", 'rb') 
       hotel1 = pickle.load(pickle_out)
       transformed_input = hotel1.transform([text])
       

       if hotel.predict(transformed_input) == 0:
        st.write(" Negative ⭐⭐⭐")

       elif    hotel.predict(transformed_input) == 1:
           st.write("Positive ⭐⭐⭐⭐⭐",icon='⭐⭐⭐⭐⭐')



# st.snow()
def ty(url):
     st.markdown(f'<p style="background-color:	tomato;color:black;font-size:24px;border-radius:2%;text-align:center">{url}</p>', unsafe_allow_html=True)
ty('♥♥THANK YOU DO VISIT AGAIN♥♥')

