import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
sentiment_analysis = pipeline('sentiment-analysis', model = model, tokenizer = tokenizer)

def reviews(input_data):
    df = pd.read_csv(input_data)
    df = df[df.Star == 1]
    df.reset_index(drop=True, inplace=True)
    X = df.Text
    sentiment = []
    for i in range(len(X)):
        if sentiment_analysis(X[i])[0]['label'] == 'POSITIVE':
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
            st.success(st.write(Filtered_Reviews))
        else:
            st.markdown('### Upload a CSV file')

    
    

if __name__ == '__main__':
    main()
