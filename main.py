import pandas as pd
import streamlit as st

import sklearn
from sklearn.metrics.pairwise import cosine_similarity


movies_dataset = pd.read_csv('final df.csv')

vectorizer = sklearn.feature_extraction.text.CountVectorizer(stop_words='english')
count_matrix = vectorizer.fit_transform(movies_dataset['document'])


cos_similarity = cosine_similarity(count_matrix, count_matrix)

indexs = pd.Series(movies_dataset.index, movies_dataset['title'])


def get_recommendation(title, cos_sim=cos_similarity):
    idx = indexs[title]
    similarity_scores = list(enumerate(cos_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:11]

    movies_idx = [ind[0] for ind in similarity_scores]
    movies = movies_dataset['title'].iloc[movies_idx]

    return movies


def main():
    st.title("Movies Recommendation System")

    try:
        title = st.text_input("- Enter the movie title:")
        if st.button("Check"):
            st.write(get_recommendation(title.lower(), cos_similarity))
    except Exception as e:
        st.write("Please make sure it is before 2016")


if __name__ == '__main__':
    main()






