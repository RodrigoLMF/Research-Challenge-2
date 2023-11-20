import pandas as pd
import numpy as np
from numpy.linalg import norm

df_content = pd.read_json("../data/content.jsonl", lines=True)
df_ratings = pd.read_json("../data/ratings.jsonl", lines=True)
df_targets = pd.read_csv("../data/targets.csv")

item_genres = {}
genres_set = set()
items = []

for index, row in df_content.iterrows():
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
user_vectors = {}
for user in user_targets:
  user_vectors[user] = np.zeros(len(genres_list))
for genre in genres_list:
  user_vectors[genre] = np.zeros(len(user_targets))


for index, row in df_target_ratings.iterrows():
  user_id = row['UserId']
  item_id = row['ItemId']
  for genre in item_genres[item_id]:
    user_vectors[user_id][genres_list.index(genre)] += 1

for user in user_targets:
  normalized_vector = (user_vectors[user]-np.min(user_vectors[user]))/(np.max(user_vectors[user])-np.min(user_vectors[user]))
  user_vectors[user] = normalized_vector

similarities = []
for index, row in df_targets.iterrows():
  user_id = row['UserId']
  item_id = row['ItemId']
  item_vector = item_vectors[item_id]
  user_vector = user_vectors[user_id]
  similarity = np.dot(item_vector, user_vector)/(norm(item_vector)*norm(user_vector))
  similarities.append(similarity)
df_targets['Similarity'] = similarities
df_targets = df_targets.sort_values(by='Similarity', ascending=False)

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
submission_df.to_csv('submission.csv', index=False)