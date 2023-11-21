import pandas as pd
import numpy as np
from numpy.linalg import norm

df_content = pd.read_json("../data/content.jsonl", lines=True)
df_ratings = pd.read_json("../data/ratings.jsonl", lines=True)
df_targets = pd.read_csv("../data/targets.csv")

item_genres = {}
genres_set = set()
items = []
item_ratings = {}

for index, row in df_content.iterrows():
    if row['imdbRating'] != 'N/A':
      item_ratings[row['ItemId']] = float(row['imdbRating'])
    else:
      item_ratings[row['ItemId']] = 5
    splitted_genres = row['Genre'].split(sep=', ')
    item_genres[row['ItemId']] = splitted_genres
    for genre in splitted_genres:
       genres_set.add(genre)
    items.append(row['ItemId'])

genres_list = list(genres_set)
item_vectors = {}
for item in items:
  item_vectors[item] = np.zeros(len(genres_list))
  for genre in item_genres[item]:
    item_vectors[item][genres_list.index(genre)] = 1

user_targets = df_targets['UserId'].unique().tolist()
df_target_ratings = df_ratings[df_ratings['UserId'].isin(user_targets)]
user_averages_by_genre = {}
user_totals_by_genre = {}
user_movies_by_genre = {}
for user in user_targets:
  user_averages_by_genre[user] = np.zeros(len(genres_list))
  user_totals_by_genre[user] = np.zeros(len(genres_list))
  user_movies_by_genre[user] = np.zeros(len(genres_list))

for index, row in df_target_ratings.iterrows():
  user_id = row['UserId']
  item_id = row['ItemId']
  for genre in item_genres[item_id]:
    user_totals_by_genre[user_id][genres_list.index(genre)] += row['Rating']
    user_movies_by_genre[user_id][genres_list.index(genre)] += 1

for user in user_targets:
  for i in range(len(genres_list)):
    if user_movies_by_genre[user][i] > 0:
      user_averages_by_genre[user][i] = user_totals_by_genre[user][i]/user_movies_by_genre[user][i] 
  normalized_vector = (user_averages_by_genre[user]-np.min(user_averages_by_genre[user]))/(np.max(user_averages_by_genre[user])-np.min(user_averages_by_genre[user]))
  user_averages_by_genre[user] = normalized_vector

predictions = []
for index, row in df_targets.iterrows():
  user_id = row['UserId']
  item_id = row['ItemId']
  item_vector = item_vectors[item_id]
  user_vector = user_averages_by_genre[user_id]
  similarity = np.dot(item_vector, user_vector)/(norm(item_vector)*norm(user_vector))
  prediction = ((10 * similarity) + item_ratings[item_id])/2
  predictions.append(prediction)
df_targets['Prediction'] = predictions
df_targets = df_targets.sort_values(by='Prediction', ascending=False)

user_prediction_items = {}
for index, row in df_targets.iterrows():
  user_id = row['UserId']
  item_id = row['ItemId']
  if user_id in user_prediction_items.keys():
    user_prediction_items[user_id].append(item_id)
  else:
    user_prediction_items[user_id] = [item_id]

final_ranking = []
for user in user_targets:
  for item in user_prediction_items[user]:
    final_ranking.append((user, item))

submission_df = pd.DataFrame(final_ranking, columns=['UserId', 'ItemId'])
submission_df.to_csv('submission3.csv', index=False)