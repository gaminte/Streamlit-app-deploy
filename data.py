import streamlit as st
import pandas as pd
from transformers import pipeline
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle

model = pickle.load(open('sentiment_analysis.sav', 'rb'))
# sentiment_analysis = pipeline('sentiment-analysis', model = 'siebert/sentiment-roberta-large-english')

def reviews(input_data):
    df = pd.read_csv(input_data)
    df = df[df.Star == 1]
    df.reset_index(drop=True, inplace=True)
    X = df.Text
    sentiment = []
    for i in range(len(X)):
        if model(X[i])[0]['label'] == 'POSITIVE':
            sentiment.append(1)
        else:
            sentiment.append(0)
    df['sentiment'] = sentiment
    y = df[df.sentiment == 1]
    y.reset_index(drop=True, inplace=True)
    return y


def main():
    st.title("Contridict Review Filter")
    input_data = st.file_uploader(label='Upload a CSV File', type=['csv'])
    
    if st.button('Filter Reviews'):
        if input_data is not None:
            Filtered_Reviews = reviews(input_data)
            st.success(Filtered_Reviews)
    
    

if __name__ == '__main__':
    main()
