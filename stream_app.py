import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Chemin de l'emplacement du fichier en local
lien = "/Users/session1/Downloads/TABLE_STREAMLIT.csv"

df = pd.read_csv(lien, sep=',')

# Fonction pour convertir le runtime en heures et minutes
def h_min(col):
    h = col // 60
    m = col % 60
    return f'{h}h {m}min'

df['runtime'] = df['runtime'].apply(h_min)

# Ajouter une colonne de résumés pour les synopsis
#df['short_overview'] = df['overview'].apply(lambda x: x[:100] + '...' if len(x) > 100 else x)

# Extraction des caractéristiques
features = pd.get_dummies(df[['genres', 'runtime']])

# Normalisation des caractéristiques (mise à l'échelle)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# On entraîne notre modèle kNN
model_knn = NearestNeighbors(metric='euclidean', algorithm='brute', n_neighbors=5)
model_knn.fit(features_scaled)

# Fonction qui retourne les films recommandés
def get_movie_recommendations(movie_title, n_recommendations=6):
    movie_idx = df[df['title_y'] == movie_title].index[0]
    distances, indices = model_knn.kneighbors(features_scaled[movie_idx].reshape(1, -1), n_neighbors=n_recommendations + 1)
    recommended_movie_indices = indices.flatten()[1:]
    recommended_movies = df.iloc[recommended_movie_indices]
    return recommended_movies

# Interface utilisateur Streamlit
st.set_page_config(page_title="Bienvenue sur notre application web", layout="wide")

# Un peu de style pour l'effet visuel
st.markdown("""
    <style>
    .main-title {
        font-size: 32px;
        color: #FAFAFA;
        text-align: center;
        margin-bottom: 40px;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .sidebar .sidebar-content h2 {
        color: #2C3E50;
        font-size: 24px;
        margin-bottom: 20px;
    }
    .sidebar .sidebar-content .section {
        margin-bottom: 20px;
    }
    .sidebar .sidebar-content .section img {
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .sidebar .sidebar-content .section p {
        color: #7F8C8D;
        margin: 0;
        font-size: 16px;
    }
    .movie-card {
        background-color: #;
        padding: 15px;
        border-radius: 10px;
        margin: 10px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: 0.3s;
    }
    .movie-card:hover {
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .movie-card img {
        border-radius: 10px;
    }
    .movie-title {
        color: #ffff66;
        font-size: 22px;
        font-weight: bold;
        margin-top: 10px;
        text-align: left;
    }
    .movie-details {
        color: #7F8C8D;
        font-size: 16px;
        text-align: left;
    }
    .button {
        display: inline-block;
        padding: 10px 20px;
        margin: 20px 0;
        font-size: 16px;
        color: white;
        background-color: #007BFF;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        text-align: center;
        text-decoration: none;
    }
    .button:hover {
        background-color: #0056b3;
    }
    .horizontal-line {
        height: 2px;
        background-color: #ccc;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# le nom du cinema avec la ligne en dessous et message de bienvenue
st.header('CINEMA EDEN', divider='rainbow')
st.markdown('<h1 class="main-title">🎬 Bienvenue sur notre application web🍿</h1>', unsafe_allow_html=True)

# Ici je mets le titre dans mon sidebar
st.sidebar.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
st.sidebar.image("/Users/session1/Downloads/images_cine-removebg-preview.png")
st.sidebar.markdown("<h2>Sélectionnez votre film</h2>", unsafe_allow_html=True)

# Selectbox pour selectionner un film avec le titre
selected_movie = st.sidebar.selectbox('', df['title_y'])

# Ajouter le bouton de recommandation pour le titre
if st.sidebar.button('Films simillaires'):

#La logique est d'afficher d'abord des films de façon aléatoire et lorsqu'on appuie sur 
# le bouton recommandé on à l'écran les films recommandés
    show_recommendations = True
    show_random_movies = False
else:
    show_recommendations = False
    show_random_movies = True

selected_movie_details = df[df['title_y'] == selected_movie].iloc[0]

st.sidebar.markdown(f"""
    <div class='section'>
        <img src="{selected_movie_details['poster_path']}" width="100%">
        <p><strong>{selected_movie_details['title_y']}</strong></p>
        <p><strong>Genre(s):</strong> {selected_movie_details['genres']}</p>
        <p><strong>Durée:</strong> {selected_movie_details['runtime']}</p>
        <p><strong>Réalisateur:</strong> {selected_movie_details['directors']}</p>
        <p><strong>Synopsis:</strong> {selected_movie_details['overview']}</p>
    </div>
    """, unsafe_allow_html=True)

#Recherche des films par genres de films
st.sidebar.markdown("<h2>Recherche par genre</h2>", unsafe_allow_html=True)
genre_to_search = st.sidebar.selectbox('', df['genres'].unique())
search_genre = st.sidebar.button('Rechercher par genre')

st.sidebar.markdown("</div>", unsafe_allow_html=True)

# Afficher les films aléatoires sur l'écran d'accueil si les recommandations ne sont pas affichées
if search_genre:
    show_random_movies = False

# Afficher les films aléatoires si `show_random_movies` est vrai
if show_random_movies:
    st.write("## Sélection de films")
    st.markdown("<hr>", unsafe_allow_html=True)
    random_movies = df.sample(9)
    cols = st.columns(3)
    for idx, (_, row) in enumerate(random_movies.iterrows()):
        with cols[idx % 3]:
            st.markdown(f""" 
                <div class="movie-card" width="8%">
                    <img src="{row['poster_path']}" width="80%">
                    <div class="movie-title">{row['title_y']}</div>
                    <div class="movie-details">
                        <p><strong>Genre(s):</strong> {row['genres']}</p>
                        <p><strong>Durée:</strong> {row['runtime']}</p>
                        <p><strong>Réalisateur(s):</strong> {row['directors']}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True) 
            st.markdown("<hr>", unsafe_allow_html=True)

# Afficher les films recommandés si `show_recommendations` est vrai
if show_recommendations:
    recommendations = get_movie_recommendations(selected_movie)
    st.write('## Vous aimerez aussi')
    st.markdown("<hr>", unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, (_, row) in enumerate(recommendations.iterrows()):
        with cols[idx % 3]:
            st.markdown(f"""
                <div class="movie-card">
                    <img src="{row['poster_path']}" width="100%">
                    <div class="movie-title">{row['title_y']}</div>
                    <div class="movie-details">
                        <p><strong>Genre(s):</strong> {row['genres']}</p>
                        <p><strong>Durée:</strong> {row['runtime']}</p>
                        <p><strong>Réalisateur(s):</strong> {row['directors']}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)

# Afficher les films du genre sélectionné si `search_genre` est vrai
if search_genre:
    genre_movies = df[df['genres'] == genre_to_search]
    st.write(f'## Film(s)du genre {genre_to_search}')
    st.markdown("<hr>", unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, (_, row) in enumerate(genre_movies.iterrows()):
        with cols[idx % 3]:
            st.markdown(f"""
                <div class="movie-card">
                    <img src="{row['poster_path']}" width="100%">
                    <div class="movie-title">{row['title_y']}</div>
                    <div class="movie-details">
                        <p><strong>Genre(s):</strong> {row['genres']}</p>
                        <p><strong>Durée:</strong> {row['runtime']}</p>
                        <p><strong>Réalisateur(s):</strong> {row['directors']}</p>
                        <p><strong>Synopsis:</strong> {row['overview']}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)
