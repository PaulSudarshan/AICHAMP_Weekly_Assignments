# Importing the required libraries
import pandas as pd
import numpy as np



# importing the dataset
movies_dataset = pd.read_csv('Dataset/ml-latest-small/movies.csv')
ratings_dataset = pd.read_csv('Dataset/ml-latest-small/ratings.csv')

# Checking the head and info of our dataset
print(movies_dataset.shape)
movies_dataset.head(5)

print(ratings_dataset.shape)
ratings_dataset.head(5)

movies_dataset.info()

ratings_dataset.info()

# sns.kdeplot(ratings_dataset['rating'])

# Converting the format of Genre column to a list and then appending to the new list
Genre = []
Genres = {}
for num in range(0, len(movies_dataset)):
    key = movies_dataset.iloc[num]['title']
    value = ' '.join(movies_dataset.iloc[num]['genres'].split('|'))
    Genres[key] = value
    Genre.append(value)

# Making a new column in our original Dataset
movies_dataset['New Genres'] = Genre
movies_dataset.head()

# movies_dataset['title'][0].split()[-1][1:-1]

# Getting the year from the movie column
years = []
for i in range(len(movies_dataset)):
    year = movies_dataset['title'][i].split()[-1][1:-1]
    years.append(year)
movies_dataset['year'] = years

movies_dataset.tail()

# ' '.join(movies_dataset['title'][0].split()[:-1])

# Deleting the year from the movies title column
movies_name = []
raw = []
for i in range(len(movies_dataset)):
    new_name = ' '.join(movies_dataset['title'][i].split()[:-1])
    movies_name.append(new_name)
movies_dataset['title'] = movies_name
movies_dataset.head()

movies_dataset.drop('genres', axis=1, inplace=True)

movies_dataset.head(2)

movies_dataset['title'] = [title.lower() for title in movies_dataset['title']]
movies_dataset['New Genres'] = [genre.lower() for genre in movies_dataset['New Genres']]
movies_dataset.head()

# Applying Feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer

tfid = TfidfVectorizer(stop_words='english')
# matrix after applying the tfidf
matrix = tfid.fit_transform(movies_dataset['New Genres'])

# Compute the cosine similarity of every genre
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(matrix, matrix)

# Making a new series which have two columns in it
# Movie name and movie id
movies_dataset = movies_dataset.reset_index()
titles = movies_dataset['title']
indices = pd.Series(movies_dataset.index, index=movies_dataset['title'])

indices.head()


# Function to make recommendation to the user
def recommendataion(movie):
    result = []
    # Getting the id of the movie for which the user want recommendation
    ind = indices[movie]
    # Getting all the similar cosine score for that movie
    sim_scores = list(enumerate(cosine_sim[ind]))
    # Sorting the list obtained
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Getting all the id of the movies that are related to the movie Entered by the user
    movie_id = [i[0] for i in sim_scores]
    print('The Movie You Should Watched Next Are --')
    print('ID ,   Name ,  Average Ratings , Year ')
    # Varible to print only top 5 movies
    count = 0
    for id in range(0, len(movie_id)):
        # to ensure that the movie entered by the user is doesnot come in his/her recommendation
        if (ind != movie_id[id]):
            ratings = ratings_dataset[ratings_dataset['movieId'] == movie_id[id]]['rating']
            avg_ratings = round(np.mean(ratings), 2)
            # To print only thoese movies which have an average ratings that is more than 3.5
            if (avg_ratings > 3.5):
                count += 1
                print(f'{movie_id[id]} , {titles[movie_id[id]]} ,{avg_ratings}')
                result.append([titles[movie_id[id]], str(avg_ratings)])
            if (count >= 5):
                break

    print('Wait!! i am telling your recommendation')
    return result


def dash_table_parser(df):
    df['Recommended Movies'] = [movie.title() for movie in df['Recommended Movies']]
    return html.Div([
        dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True, dark=False, className='text-field-result')
    ])


import base64
import datetime
import io

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash_table.Format import Format, Group, Scheme, Symbol
import dash_bootstrap_components as dbc
import plotly.express as px
from plotly.tools import mpl_to_plotly

image_bg = 'assets//bg.jpg'
encoded_bg = base64.b64encode(open(image_bg, 'rb').read())

# image_logo = 'D://Work Files//Einnel//submission//assets//logo.png' # replace with your own image
# encoded_logo = base64.b64encode(open(image_logo, 'rb').read())

app = dash.Dash(external_stylesheets=[dbc.themes.SLATE])
colors = {'background': '#01151a', 'text': '#beeefa'}

app.layout = html.Div([
    html.H1('Movie Recommendation System', className='text-field-title'),
    dbc.Input(id="movie-name", placeholder="Enter Name of Movie", type="text",
              style=dict(width='20%', display='list-item'), className='text-field'),
    dbc.Button("Submit", id='submit', color="success", size="lg", n_clicks=0, className="home-button"),
    html.Div(id='output'),

    html.Div(children=[
        html.Img(className='bg', src='data:image/png;base64,{}'.format(encoded_bg.decode()))])

])


# ===================================================================================================#

@app.callback(Output('output', 'children'),
              [Input('submit', 'n_clicks')],
              [State('movie-name', 'value')])
#              [State('input-on-submit', 'value')])

def update_datatable(n_clicks, movie):
    if n_clicks:
        df = pd.DataFrame(recommendataion(movie.lower()), columns=['Recommended Movies', 'Rating'])
        return [dash_table_parser(df)]


if __name__ == '__main__':
    app.run_server(debug=False)






