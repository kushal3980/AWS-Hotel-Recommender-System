
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer



df = pd.read_csv('C:\\Users\\Kushal\\Desktop\\daiict\\VAC\\endtoend\\VAC_PROJECT\\Datafiniti_Hotel_Reviews.csv') ### Make changes here



df = df[['id','name','reviews.rating','reviews.username','city','country','province', 'reviews.text']]
df.rename(columns = {'reviews.username':'username','reviews.rating':'rating','reviews.text':'text'}, inplace = True)

df.dropna(subset=['username', 'name', 'text', 'rating'], inplace=True)
df['text'] = df['text'].str.lower().str.replace('[^\w\s]', '')  # Lowercase and remove punctuation

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

df['text'] = df['text'].apply(preprocess_text)



from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['text'])

import scipy.sparse
scipy.sparse.save_npz('tfidf_matrix.npz', tfidf_matrix)

user_profiles = df.groupby('username')['text'].apply(lambda x: ' '.join(x)).reset_index()
user_profiles['text'] = user_profiles['text'].apply(preprocess_text)
user_tfidf_matrix = tfidf.transform(user_profiles['text'])

user_indices = pd.Series(user_profiles.index, index=user_profiles['username']).drop_duplicates()


from sklearn.metrics.pairwise import cosine_similarity

cosine_sim_users = cosine_similarity(user_tfidf_matrix, tfidf_matrix)

hotel_indices = pd.Series(df.index, index=df['name']).drop_duplicates()




def get_recommendations(city, username, cosine_sim_users=cosine_sim_users):
    city_hotels = df[df['city'].str.lower() == city.lower()]

    if city_hotels.empty:
        return "No hotels found in this city."

    if username not in user_indices:
        return "User not found."
    user_idx = user_indices[username]

    sim_scores = list(enumerate(cosine_sim_users[user_idx]))

    city_indices = city_hotels.index
    sim_scores = [score for score in sim_scores if score[0] in city_indices]

    hotel_sim_scores = {}
    for idx, score in sim_scores:
        hotel_name = df.iloc[idx]['name']
        rating = df.iloc[idx]['rating']
        if hotel_name in hotel_sim_scores:
            hotel_sim_scores[hotel_name] += score * rating
        else:
            hotel_sim_scores[hotel_name] = score * rating

    sorted_hotels = sorted(hotel_sim_scores.items(), key=lambda x: x[1], reverse=True)

    top_hotels = [hotel for hotel, score in sorted_hotels[:10]]

    return top_hotels

recommendations = get_recommendations('New York', 'Ron')
#print(recommendations)



from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/recommend', methods=['GET'])
def recommend():
    city = request.args.get('city')
    username = request.args.get('username')
    
    if not city or not username:
        return jsonify({"error": "City and username parameters are required"}), 400
    
    recommendations = get_recommendations(city, username)
    return jsonify({"recommendations": recommendations})

if __name__ == '__main__':
    app.run(debug=True)



