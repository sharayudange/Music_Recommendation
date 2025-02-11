from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the data and model
data = pickle.load(open(r"C:\Users\Administrator\Desktop\Music_Recomm\data.pkl", 'rb'))
song_cluster_pipeline = pickle.load(open(r"C:\Users\Administrator\Desktop\Music_Recomm\song_cluster_pipeline.pkl", 'rb'))

from sklearn.metrics.pairwise import cosine_similarity

def get_similarities(song_name, data):
    num_array1 = data[data['name'] == song_name].select_dtypes(include='number').to_numpy()
    input_song_cluster = data[data['name'] == song_name]['cluster_label'].iloc[0]
    cluster_data = data[data['cluster_label'] == input_song_cluster]
    sim = []
    for idx, row in cluster_data.iterrows():
        name = row['name']
        num_array2 = data[data['name'] == name].select_dtypes(include='number').to_numpy()
        num_sim = cosine_similarity(num_array1, num_array2)[0][0]
        sim.append((name, row['artists'], num_sim))
    return sorted(sim, key=lambda x: x[2], reverse=True)[1:11]

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        song_name = request.form['song_name']
        if data[data['name'] == song_name].shape[0] > 0:
            recommendations = get_similarities(song_name, data)
        else:
            recommendations = [('Invalid song name or song not found', '', 0)]
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
