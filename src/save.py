import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Obtendo o caminho para a pasta 'data' e os arquivos dentro dela
data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
ratings_path = os.path.join(data_path, 'ratings.jsonl')
content_path = os.path.join(data_path, 'content.jsonl')
targets_path = os.path.join(data_path, 'targets.csv')

# Leitura dos dados
ratings = pd.read_json(ratings_path, lines=True)
content = pd.read_json(content_path, lines=True)
targets = pd.read_csv(targets_path)

# Representação dos itens (filmes) usando os gêneros
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
item_tfidf_matrix = tfidf_vectorizer.fit_transform(content['Genre'].fillna(''))
print(f'item_tfidf_matrix shape: {item_tfidf_matrix.shape}')

# Representação dos usuários usando os gêneros dos filmes que avaliaram
user_ratings = ratings.merge(content, how='left', on='ItemId')
user_tfidf_matrix = tfidf_vectorizer.transform(user_ratings.groupby('UserId')['Genre'].apply(lambda x: ' '.join(x)).fillna(''))
print(f'user_tfidf_matrix shape: {user_tfidf_matrix.shape}')

# Aplicação do algoritmo de Rocchio
alpha = 1.0
beta = 0.5

user_ratings_grouped = user_ratings.groupby('UserId')
user_ratings_mean = user_ratings_grouped['Rating'].mean()

user_profile = alpha * user_tfidf_matrix.T @ user_ratings_mean - beta * np.mean(user_ratings_mean)

user_profile = user_profile.reshape(1, -1)

# Geração de recomendações para cada usuário
results = []
for user_id, group in targets.groupby('UserId'):
    # Calcular a similaridade entre o perfil do usuário e os itens
    relevance_scores = cosine_similarity(user_profile, item_tfidf_matrix)

    # Obter os índices dos filmes recomendados em ordem decrescente de relevância
    recommended_indices = np.argsort(relevance_scores[0])[::-1][:100]

    # Obter os IDs dos filmes recomendados
    recommended_movies = content.iloc[recommended_indices]['ItemId'].values

    # Adicionar diversificação?

    # Armazenar os resultados
    results.extend([(user_id, movie_id) for movie_id in recommended_movies])

# Criar um DataFrame com os resultados
submission_df = pd.DataFrame(results, columns=['UserId', 'ItemId'])

# Salvar os resultados em um arquivo CSV
submission_df.to_csv('submission.csv', index=False)