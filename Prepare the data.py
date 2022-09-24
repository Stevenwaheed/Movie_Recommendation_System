import pandas as pd
import numpy as np

from ast import literal_eval


def production_companies(companies):
    for details in companies:
        return details["name"]


def get_info(column):
    items = 3
    details_list = []
    while items > 0:
        details_list = [details["name"] for details in column]
        items = items - 1
    return details_list


def director(crew):
    for details in crew:
        if details['job'] == "Director":
            return details["name"]
    return np.nan


movies_df = pd.read_csv('tmdb_5000_movies.csv')
credits_df = pd.read_csv('tmdb_5000_credits.csv')

credits_df.rename(columns={'movie_id': 'id'}, inplace=True)

movies_data = movies_df[['id', 'genres', 'overview', 'keywords', 'production_companies', 'tagline']]

movies_data = movies_data.merge(credits_df, on="id")

movies_data.drop(columns='id', inplace=True)


dict_columns = ['genres', 'keywords', 'production_companies', 'cast', 'crew']
for column in dict_columns:
    movies_data[column] = movies_data[column].apply(literal_eval)


copy_movies_data = movies_data.copy()

copy_movies_data['production_companies'] = movies_data['production_companies'].apply(production_companies)

copy_movies_data['director'] = movies_data['crew'].apply(director)


cols = ['genres', 'keywords', 'cast']
for col in cols:
    copy_movies_data[col] = movies_data[col].apply(get_info)


for col in cols:
    copy_movies_data = copy_movies_data[copy_movies_data[col].map(lambda d: len(d)) > 0]


copy_movies_data.dropna(inplace=True)

copy_movies_data['title'] = movies_data['title'].apply(lambda row: row.lower())


def create_doc(df):
    return ' '.join(df.genres) + ' ' + ' '.join(
        df.keywords) + ' ' + df.production_companies + ' ' + df.tagline + ' ' + ' '.join(
        df.cast) + ' ' + df.director + ' ' + df.overview


copy_movies_data['document'] = copy_movies_data.apply(create_doc, axis=1)

copy_movies_data.to_csv('final df.csv')
