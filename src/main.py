import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Obtendo o caminho para a pasta 'data' e os arquivos dentro dela
print('Obtendo o caminho para a pasta data e os arquivos dentro dela')
data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
ratings_path = os.path.join(data_path, 'ratings.jsonl')
content_path = os.path.join(data_path, 'content.jsonl')
targets_path = os.path.join(data_path, 'targets.csv')

# Leitura dos dados
print('Leitura dos dados')
ratings = pd.read_json(ratings_path, lines=True)
content = pd.read_json(content_path, lines=True)
targets = pd.read_csv(targets_path)

# Manipulação dos valores NaN para imdbRating e imdbVotes
print('Manipulação dos valores NaN para imdbRating e imdbVotes')
content['imdbRating'] = content['imdbRating'].apply(lambda x: float(x) if x != 'N/A' else np.nan)
content['imdbVotes'] = content['imdbVotes'].apply(lambda x: int(x.replace(',', '')) if x != 'N/A' else 0)

# Representação dos itens (filmes) usando gêneros e características do IMDb
print('Representação dos itens (filmes) usando gêneros e características do IMDb')
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
item_tfidf_matrix_genre = tfidf_vectorizer.fit_transform(content['Genre'].fillna(''))
item_imdb_features = content[['imdbRating', 'imdbVotes']].fillna(0).values
item_tfidf_matrix = np.hstack((item_tfidf_matrix_genre.toarray(), item_imdb_features))

# Representação dos usuários usando os gêneros dos filmes que avaliaram
print('Representação dos usuários usando os gêneros dos filmes que avaliaram')
user_ratings = ratings.merge(content, how='left', on='ItemId')
user_tfidf_matrix_genre = tfidf_vectorizer.transform(user_ratings.groupby('UserId')['Genre'].apply(lambda x: ' '.join(x)).fillna(''))
user_imdb_features = user_ratings.groupby('UserId')[['imdbRating', 'imdbVotes']].mean().fillna(0).values
user_tfidf_matrix = np.hstack((user_tfidf_matrix_genre.toarray(), user_imdb_features))

# Aplicação do algoritmo de Rocchio
print('Aplicação do algoritmo de Rocchio')
alpha = 1.0
beta = 0.5
gamma = 0.2  

# Calcular o vetor de perfil do usuário 
print('Calcular o vetor de perfil do usuário usando o algoritmo de Rocchio')
user_ratings_grouped = user_ratings.groupby('UserId')
user_ratings_mean = user_ratings_grouped['Rating'].mean()

# Vetor de consulta original
print('Vetor de consulta original')
user_profile_genre = alpha * user_tfidf_matrix_genre.T @ user_ratings_mean

# Vetores de documentos relacionados e não relacionados
print('Vetores de documentos relacionados e não relacionados')
DR = user_ratings_grouped.filter(lambda x: x['Rating'].max() >= 4)['ItemId'].values
DNR = user_ratings_grouped.filter(lambda x: x['Rating'].max() < 4)['ItemId'].values

# Vetor de consulta modificado usando o algoritmo de Rocchio
print('Vetor de consulta modificado usando o algoritmo de Rocchio')
user_profile_genre_mod = (
    alpha * user_profile_genre +
    beta / len(DR) * np.sum(item_tfidf_matrix[np.isin(content['ItemId'], DR)], axis=0) -
    gamma / len(DNR) * np.sum(item_tfidf_matrix[np.isin(content['ItemId'], DNR)], axis=0)
)

# Combinar os vetores de perfil do usuário
print('Combinar os vetores de perfil do usuário')
user_profile = np.hstack((user_profile_genre_mod, user_imdb_features))
user_profile = user_profile.reshape(1, -1)

# Substituir NaN por 0 nas matrizes
print('Substituir NaN por 0 nas matrizes')
item_tfidf_matrix = np.nan_to_num(item_tfidf_matrix)
user_tfidf_matrix = np.nan_to_num(user_tfidf_matrix)

# Geração de recomendações para cada usuário
print('Geração de recomendações para cada usuário')
results = []
for user_id, group in targets.groupby('UserId'):
    # Calcular a similaridade entre o perfil do usuário e os itens
    relevance_scores = cosine_similarity(user_profile, item_tfidf_matrix)

    # Obter os índices dos filmes recomendados em ordem decrescente de relevância
    recommended_indices = np.argsort(relevance_scores[0])[::-1][:100]

    # Obter os IDs dos filmes recomendados
    recommended_movies = content.iloc[recommended_indices]['ItemId'].values

    # Armazenar os resultados
    results.extend([(user_id, movie_id) for movie_id in recommended_movies])

# Criar um DataFrame com os resultados
print('Criar um DataFrame com os resultados')
submission_df = pd.DataFrame(results, columns=['UserId', 'ItemId'])

# Salvar os resultados em um arquivo CSV
print('Salvar os resultados em um arquivo CSV')
submission_df.to_csv('submission.csv', index=False)
