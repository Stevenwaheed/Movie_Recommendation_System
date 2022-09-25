import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


movies_dataset = pd.read_csv('final df.csv')

vectorizer = CountVectorizer(stop_words='english')
count_matrix = vectorizer.fit_transform(movies_dataset['document'])


cos_similarity = cosine_similarity(count_matrix, count_matrix)

indexs = pd.Series(movies_dataset.index, movies_dataset['title'])


def get_all_titles(new_title):
  title_list = []
  for index in indexs.index:
    if new_title in index or new_title == index:
      title_list.append(index)

  return title_list


def get_recommendation(title, cos_sim=cos_similarity):
    title_list = get_all_titles(title)

    all_similar_movies = []
    all_similar_scores = []

    for title in title_list:
      idx = indexs[title]
      similarity_scores = list(enumerate(cos_sim[idx]))

      # sort the scores descending
      similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

      # select the top 10 scores
      similarity_scores = similarity_scores

      # extract these scores' indices
      movies_idx = [ind[0] for ind in similarity_scores]

      # extract the movie's title depend on movie's index
      movies = movies_dataset['title'].iloc[movies_idx]
      
      all_similar_movies.extend(movies)
      all_similar_scores.extend(similarity_scores)

    return all_similar_movies, all_similar_scores
    


def Extract_Similarity_Scores(similarity_scores):
  score_list = []

  for score in similarity_scores:
    score_list.append(score[1])
  
  return score_list


def main():
    st.title("Movies Recommendation System")
    st.write('- This system uses "Cosine Similarity Algorithm" to compare the user\'s movie with the dataset and recommends some movies based on the similarity.')
#     try:
    title = st.text_input("- Enter the movie title:")
    if st.button("Check"):
        movie, scores = get_recommendation(title.lower(), cos_similarity)
        Similarity_Scores = Extract_Similarity_Scores(scores)

        mixture_movies_scores = pd.Series(Similarity_Scores, movie)

        movie_data_frame = pd.DataFrame(mixture_movies_scores)

        st.dataframe(movie_data_frame.head(20)[1:], 1000, 1000)
            
#     except Exception as e:
#         st.write("Please make sure it is before 2016")

if __name__ == '__main__':
    main()






