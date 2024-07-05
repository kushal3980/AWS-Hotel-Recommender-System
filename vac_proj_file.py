import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse
from flask import Flask, render_template, redirect, url_for, request, jsonify, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Length, EqualTo

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=20)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')

df = pd.read_csv('C:\\Users\\Kushal\\Desktop\\daiict\\VAC\\endtoend\\VAC_PROJECT\\Datafiniti_Hotel_Reviews.csv')

df.dropna(subset=['username', 'name', 'text', 'rating'], inplace=True)
df['text'] = df['text'].str.lower().str.replace('[^\w\s]', '')  # Lowercase and remove punctuation

nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))
ps = nltk.stem.PorterStemmer()

def preprocess_text(text):
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

df['text'] = df['text'].apply(preprocess_text)

tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['text'])

scipy.sparse.save_npz('C:\\Users\\Kushal\\Desktop\\daiict\\VAC\\endtoend\\VAC_PROJECT\\.npz', tfidf_matrix)
with open('tfidf_vectorizer.pkl', 'wb') as f:
    import pickle
    pickle.dump(tfidf, f)

user_profiles = df.groupby('username')['text'].apply(lambda x: ' '.join(x)).reset_index()
user_profiles['text'] = user_profiles['text'].apply(preprocess_text)
user_tfidf_matrix = tfidf.transform(user_profiles['text'])

scipy.sparse.save_npz('C:\\Users\\Kushal\\Desktop\\daiict\\VAC\\endtoend\\VAC_PROJECT\\.npz', user_tfidf_matrix)

user_profiles.to_csv('C:\\Users\\Kushal\\Desktop\\daiict\\VAC\\endtoend\\VAC_PROJECT\\user_profiles.csv', index=False)






from flask import Flask, render_template, redirect, url_for, request, jsonify, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import scipy.sparse
import pickle

app = Flask(__name__)
app.secret_key = 'your_secret_key'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

df = pd.read_csv('C:\\Users\\Kushal\\Desktop\\daiict\\VAC\\endtoend\\VAC_PROJECT\\Datafiniti_Hotel_Reviews.csv')
users = df['username'].unique().tolist()

class User(UserMixin):
    def __init__(self, username):
        self.id = username

@login_manager.user_loader
def load_user(user_id):
    if user_id in users:
        return User(user_id)
    return None

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegistrationForm()  # Create an instance of the RegistrationForm class
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        # Check if the username already exists
        if username in users:
            flash('Username already exists. Please choose a different one.', 'error')
        else:
            # Add the new user to the list of users
            users.append(username)
            flash('Your account has been created! You can now log in.', 'success')
            return redirect(url_for('login'))
    return render_template('signup.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and password == 'Password':
            user = User(username)
            login_user(user)
            return redirect(url_for('city_input'))
        else:
            flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))



@app.route('/city_input', methods=['GET', 'POST'])
@login_required
def city_input():
    if request.method == 'POST':
        city = request.form['city']
        return redirect(url_for('recommend', city=city))
    return render_template('city_input.html')


@app.route('/recommend', methods=['GET'])
@login_required
def recommend():
    '''
    city = request.args.get('city')
    if not city:
        return jsonify({"error": "City parameter is required"}), 400

    username = current_user.id
    recommendations = get_recommendations(city, username)
    return jsonify({"recommendations": recommendations})
    '''
    city = request.args.get('city')
    if not city:
        return jsonify({"error": "City parameter is required"}), 400
    username = current_user.id
    recommendations = get_recommendations(city, username)
    return render_template('recommend.html', hotels=recommendations)


@app.route('/hotel/<hotel_name>', methods=['GET'])
@login_required
def hotel_details(hotel_name):
    hotel = df[df['name'] == hotel_name].iloc[0]
    reviews = df[df['name'] == hotel['name']][['username', 'rating', 'text']].to_dict(orient='records')
    return render_template('hotel_details.html', hotel=hotel, reviews=reviews)
'''
@app.route('/hotel/<str:hotel_name>',methods=['GET'])
#@login_required
def hotel_details(hotel_name):
    
    return render_template('hotel_details.html', hotel=df[hotel_name])
'''   




@app.route('/hotel/<hotel_name>/review', methods=['POST'])
@login_required
def submit_review(hotel_name):
    df = pd.read_csv('C:\\Users\\Kushal\\Desktop\\daiict\\VAC\\endtoend\\VAC_PROJECT\\Datafiniti_Hotel_Reviews.csv')
    rating = request.form['rating']
    review_text = request.form['text']
    
    new_review = pd.DataFrame({
        'id': [len(df) + 1],
        'name': [df[df['name'] == hotel_name]['name'].values[0]],
        'rating': [rating],
        'username': [current_user.id],
        'city': [df[df['name'] == hotel_name]['city'].values[0]],
        'country': [df[df['name'] == hotel_name]['country'].values[0]],
        'province': [df[df['name'] == hotel_name]['province'].values[0]],
        'text': [review_text]
    })
    #global df
    
    df = df.append(new_review, ignore_index=True)
    
    df.to_csv('C:\\Users\\Kushal\\Desktop\\daiict\\VAC\\endtoend\\VAC_PROJECT\\Datafiniti_Hotel_Reviews.csv', index=False)
    
    return redirect(url_for('hotel_details', hotel_name=hotel_name))

def get_recommendations(city, username):
    global df
    df = pd.read_csv('C:\\Users\\Kushal\\Desktop\\daiict\\VAC\\endtoend\\VAC_PROJECT\\Datafiniti_Hotel_Reviews.csv')
    tfidf_matrix = scipy.sparse.load_npz('C:\\Users\\Kushal\\Desktop\\daiict\\VAC\\endtoend\\VAC_PROJECT\\.npz')
    with open('C:\\Users\\Kushal\\Desktop\\daiict\\VAC\\endtoend\\VAC_PROJECT\\tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    user_tfidf_matrix = scipy.sparse.load_npz('C:\\Users\\Kushal\\Desktop\\daiict\\VAC\\endtoend\\VAC_PROJECT\\.npz')
    user_profiles = pd.read_csv('C:\\Users\\Kushal\\Desktop\\daiict\\VAC\\endtoend\\VAC_PROJECT\\user_profiles.csv')
    user_indices = pd.Series(user_profiles.index, index=user_profiles['username']).drop_duplicates()

    city_hotels = df[df['city'].str.lower() == city.lower()]

    if city_hotels.empty:
        return "No hotels found in this city."

    if username not in user_indices:
        top_hotels = city_hotels.sort_values(by='rating', ascending=False)['name'].unique()[:10]
        return top_hotels

    user_idx = user_indices[username]

    cosine_sim_users = cosine_similarity(user_tfidf_matrix, tfidf_matrix)

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

if __name__ == '__main__':
    df = pd.read_csv('C:\\Users\\Kushal\\Desktop\\daiict\\VAC\\endtoend\\VAC_PROJECT\\Datafiniti_Hotel_Reviews.csv')
    app.run(debug=True)
