
from flask import Flask, render_template, request
import pandas as pd
import pickle
import requests
from difflib import get_close_matches
import os
import gdown
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

GDRIVE_FILES = {
    "similarity_matrix": "16QMPz4Av1JCCbBJqSe3rwVng6qv3PV9H",
    "movie_dict": "1h9jiL7sbcbQXG0Lc5IYRfg3bndKDUny5"
}

LOCAL_FILES = {
    "similarity_matrix": "similarity_matrix.pkl",
    "movie_dict": "movie_dict1.pkl"
}

# Download a file from Google Drive if it doesn't exist locally
def download_file_from_drive(file_id, destination):
    if not os.path.exists(destination):
        logging.info(f"Downloading {destination} from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", destination, quiet=False)

# Download all required files from Google Drive
for key, file_id in GDRIVE_FILES.items():
    download_file_from_drive(file_id, LOCAL_FILES[key])

# Load movie data and similarity model
try:
    with open(LOCAL_FILES["movie_dict"], "rb") as file:
        movies_dict = pickle.load(file)
    movies = pd.DataFrame(movies_dict)
    logging.info("Successfully loaded movie data")
    
    with open(LOCAL_FILES["similarity_matrix"], "rb") as file:
        similarity = pickle.load(file)
    logging.info("Successfully loaded similarity matrix")
    
except Exception as e:
    logging.error(f"Error loading data files: {str(e)}")
    raise SystemExit("Failed to load required data files")

# Function to fetch movie posters with error handling
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get("poster_path")
        
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"
        return "https://via.placeholder.com/500x750?text=Poster+Not+Available"
    
    except Exception as e:
        logging.error(f"Error fetching poster for movie ID {movie_id}: {str(e)}")
        return "https://via.placeholder.com/500x750?text=Error+Loading+Poster"

# Movie recommendation function with improved error handling
def recommend(movie_name):
    try:
        logging.debug(f"Received recommendation request for: {movie_name}")
        
        close_matches = get_close_matches(
            movie_name.lower(),
            [title.lower() for title in movies["title"]],
            n=5,
            cutoff=0.5
        )
        
        if not close_matches:
            logging.warning(f"No close matches found for: {movie_name}")
            return None, []

        # Get original case match from dataframe
        original_match = movies[movies["title"].str.lower() == close_matches[0]].iloc[0]["title"]
        movie_index = movies[movies["title"] == original_match].index[0]
        
        distances = similarity[movie_index]
        recommended_indices = sorted(
            enumerate(distances),
            key=lambda x: x[1],
            reverse=True
        )[1:6]

        recommendations = []
        for idx, score in recommended_indices:
            try:
                movie_data = movies.iloc[idx]
                recommendations.append({
                    "title": movie_data.title,
                    "poster": fetch_poster(movie_data.movie_id),
                    "year": movie_data.get("year", "N/A"),
                    "score": float(score)
                })
            except Exception as e:
                logging.error(f"Error processing recommendation index {idx}: {str(e)}")

        logging.debug(f"Generated {len(recommendations)} recommendations")
        return original_match, recommendations

    except Exception as e:
        logging.error(f"Recommendation error: {str(e)}")
        return None, []

# Route for home page with enhanced error handling
@app.route("/", methods=["GET", "POST"])
def home():
    try:
        recommendations = []
        match = None
        error = None

        if request.method == "POST":
            movie_name = request.form.get("movie_name", "").strip()
            if not movie_name:
                error = "Please enter a movie name"
            else:
                match, recommendations = recommend(movie_name)
                if not match:
                    error = f"No close match found for '{movie_name}'. Try being more specific."
                
        return render_template(
            "index.html",
            match=match,
            recommendations=recommendations,
            error=error,
            movies=movies["title"].tolist()
        )

    except Exception as e:
        logging.error(f"Route error: {str(e)}")
        return render_template("error.html", error="An unexpected error occurred"), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
