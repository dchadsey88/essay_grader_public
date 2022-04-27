import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Wordcloud
from wordcloud import WordCloud

# import joblib
import pickle

import os
import sys
path = os.path.abspath('.') + "/essay_grader/"
sys.path.append(path)

from essay_grader.utils import plot_feature
from essay_grader.text_feature import gen_text_feature


# Load Our Model
@st.cache
def load_prediction_models(model_file):
    loaded_models = joblib.load(open(os.path.join(model_file),"rb"))
    return loaded_models

@st.cache
def load_df():
    with open("./essay_grader/pickle_data/text_feature_df2.pkl", "rb") as file:
        feature_df = pickle.load(file)
    return feature_df


def show_predict(text_title, text_body):
    try:
        if len(text_body) < 500:
            return 1
        small_data = {"text":text_body,"title_name":text_title}
        small_df = pd.DataFrame(small_data,index=[0])
        final_df =gen_text_feature(small_df)
        #load model
        rf_model = pickle.load(open('./essay_grader/pickle_data/rf_model.pkl', 'rb'))
        predict = rf_model.predict(final_df)[0]
        return predict
    except:
        return 0




def main():
    st.title('IB test TOK Essay Grader')

    image = Image.open('./essay_grader/data/tok_images.jpeg')
    st.image(image, use_column_width=True)

    st.header("What is TOK?")
    st.markdown("Theory of knowledge (TOK) plays a special role in the \
          International Baccalaureate® (IB) Diploma Programme (DP), \
          by providing an opportunity for students to reflect on the \
          nature of knowledge, and on how we know what we claim to know.")

    activities = ["Predict","Analyze"]

    choice = st.sidebar.selectbox("CHOOSE", activities)

    if choice =="Predict":
        text_title = st.text_area("Enter Title","Type Here",max_chars=300)
        text_body = st.text_area("Enter Essay","Type Here")
        #show wordcloud
        if st.checkbox("Wordcloud"):
            wordcloud =  WordCloud().generate(text_body)
            plt.imshow(wordcloud,interpolation='bilinear')
            plt.axis("off")
            st.pyplot()

        eva_btn = st.button("Evaluate")


        if eva_btn:
            st.write("show predict result")
            st.text("Original title :\n{}".format(text_title))
            st.text("Original essay :\n{}".format(text_body))


            final_result = show_predict(text_title,text_body)
            st.success("Essay evaluated as Score around {}".format(final_result))



        btn = st.button("Celebrate!")


        if btn:
            st.balloons()



    if choice =="Analyze":
        st.info("Essay Analyze")

        features = ['Show DataFrame','vocab_richness', 'mean_word_syllable', 'word_count', 'sentence_count',
       'avg_sentence_length', 'count_stopwords', 'flesch_reading_ease',
       'freq_wok_words', 'freq_aok_words', 'freq_cliche_words',
       'freq_argument_words', 'freq_absolute_words',
        ]

        task_choice = st.selectbox("Choose Analyze Feature",features)
        df = load_df()
        if st.button("Analyze"):
            if task_choice == 'Show DataFrame':
                st.dataframe(df.head())
            else:
                # st.markdown("show plot")
                fig = plot_feature(df.score, df[task_choice])
                st.pyplot(fig)



if __name__ == '__main__':
    main()


