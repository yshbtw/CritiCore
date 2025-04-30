from flask import Flask, render_template, request, jsonify, redirect, url_for
import pickle
import pandas as pd
import requests
import os
from dotenv import load_dotenv
from functools import lru_cache
import traceback
import logging
from flask_caching import Cache
import json
import re

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure cache
cache_config = {
    "DEBUG": True,
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": 3600  # 1 hour
}
app.config.from_mapping(cache_config)
cache = Cache(app)

# TMDB API configuration
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "YOUR_API_KEY")
TMDB_ACCESS_TOKEN = os.getenv("TMDB_ACCESS_TOKEN", "YOUR_ACCESS_TOKEN")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/"

# Headers for TMDB API requests
TMDB_HEADERS = {
    "accept": "application/json",
    "Authorization": f"Bearer {TMDB_ACCESS_TOKEN}"
}

# Load movie data
try:
    logger.info("Loading movie data from Movie_list.pkl...")
    movies = pickle.load(open('Movie_list.pkl', 'rb'))
    movie_list = movies['title'].tolist()
    logger.info(f"Loaded {len(movie_list)} movies.")
    logger.info(f"Movie DataFrame columns: {movies.columns.tolist()}")
    logger.info(f"First movie: {movies.iloc[0]}")
except Exception as e:
    logger.error(f"Error loading movie list: {e}")
    traceback.print_exc()
    movies = pd.DataFrame()
    movie_list = []

# Load KNN model
try:
    logger.info("Loading KNN model from knn_model.pkl...")
    knn_model = pickle.load(open('knn_model.pkl', 'rb'))
    logger.info("KNN model loaded.")
    if hasattr(knn_model, '_fit_X'):
        logger.info(f"KNN model _fit_X shape: {knn_model._fit_X.shape}")
    else:
        logger.warning("KNN model doesn't have _fit_X attribute!")
except Exception as e:
    logger.error(f"Error loading KNN model: {e}")
    traceback.print_exc()
    knn_model = None

def format_runtime(minutes):
    """Format minutes into hours and minutes."""
    if not minutes or minutes == 0:
        return "N/A"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m" if hours > 0 else f"{mins}m"

def clean_movie_title(title):
    """Clean movie title for URL usage."""
    cleaned = re.sub(r'[^\w\s-]', '', title)
    cleaned = re.sub(r'[-\s]+', '-', cleaned).strip('-_')
    return cleaned.lower()

@lru_cache(maxsize=100)
def search_movie(query):
    """Search for a movie by title."""
    try:
        logger.info(f"Searching for movie: {query}")
        url = f"{TMDB_BASE_URL}/search/movie"
        params = {
            "query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": 1
        }
        
        response = requests.get(url, headers=TMDB_HEADERS, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get("results"):
            logger.info(f"Found {len(data['results'])} results for query: {query}")
            return data["results"]
        else:
            logger.warning(f"No results found for query: {query}")
            return []
    except Exception as e:
        logger.error(f"Error searching for movie {query}: {e}")
        traceback.print_exc()
        return []

@lru_cache(maxsize=1000)
@cache.memoize(timeout=86400)  # Cache for 24 hours
def fetch_movie_details(movie_id):
    """Fetch detailed information about a movie."""
    try:
        logger.info(f"Fetching details for movie ID: {movie_id}")
        
        # Get movie details
        url = f"{TMDB_BASE_URL}/movie/{movie_id}"
        params = {
            "append_to_response": "videos,credits,keywords,recommendations,reviews",
            "language": "en-US"
        }
        
        response = requests.get(url, headers=TMDB_HEADERS, params=params)
        response.raise_for_status()
        movie_data = response.json()
        
        # Extract cast and crew
        cast = movie_data.get("credits", {}).get("cast", [])
        crew = movie_data.get("credits", {}).get("crew", [])
        
        # Find directors
        directors = [person['name'] for person in crew if person.get('job') == 'Director']
        
        # Get trailer
        trailer = None
        videos = movie_data.get("videos", {}).get("results", [])
        trailer_videos = [v for v in videos if v.get("type") == "Trailer" and v.get("site") == "YouTube"]
        if trailer_videos:
            trailer = f"https://www.youtube.com/embed/{trailer_videos[0]['key']}"
        
        # Get poster and backdrop paths
        poster_path = movie_data.get("poster_path")
        backdrop_path = movie_data.get("backdrop_path")
        poster_url = f"{TMDB_IMAGE_BASE_URL}w500{poster_path}" if poster_path else None
        backdrop_url = f"{TMDB_IMAGE_BASE_URL}original{backdrop_path}" if backdrop_path else None
        
        # Extract genres
        genres = [genre["name"] for genre in movie_data.get("genres", [])]
        
        # Extract keywords
        keywords = [keyword["name"] for keyword in movie_data.get("keywords", {}).get("keywords", [])]
        
        # Create movie details object
        details = {
            "id": movie_data.get("id"),
            "title": movie_data.get("title"),
            "original_title": movie_data.get("original_title"),
            "poster": poster_url,
            "backdrop": backdrop_url,
            "rating": movie_data.get("vote_average"),
            "votes": movie_data.get("vote_count"),
            "plot": movie_data.get("overview", "No plot available"),
            "tagline": movie_data.get("tagline"),
            "year": movie_data.get("release_date", "")[:4] if movie_data.get("release_date") else "N/A",
            "genres": genres,
            "runtime": format_runtime(movie_data.get("runtime")),
            "runtime_minutes": movie_data.get("runtime"),
            "directors": directors,
            "cast": [person["name"] for person in cast[:10]],
            "full_cast": [{"name": person["name"], "character": person.get("character", ""), 
                          "profile_path": f"{TMDB_IMAGE_BASE_URL}w185{person['profile_path']}" if person.get("profile_path") else None} 
                          for person in cast[:20]],
            "trailer": trailer,
            "keywords": keywords,
            "budget": movie_data.get("budget"),
            "revenue": movie_data.get("revenue"),
            "imdb_id": movie_data.get("imdb_id"),
            "popularity": movie_data.get("popularity"),
            "status": movie_data.get("status"),
            "release_date": movie_data.get("release_date"),
            "recommendations": [{
                "id": rec["id"],
                "title": rec["title"],
                "poster": f"{TMDB_IMAGE_BASE_URL}w500{rec['poster_path']}" if rec.get("poster_path") else None,
                "rating": rec.get("vote_average"),
                "year": rec.get("release_date", "")[:4] if rec.get("release_date") else "N/A",
            } for rec in movie_data.get("recommendations", {}).get("results", [])[:10]],
            "reviews": [{
                "author": rev["author"],
                "content": rev["content"],
                "rating": rev.get("author_details", {}).get("rating")
            } for rev in movie_data.get("reviews", {}).get("results", [])[:5]]
        }
        
        logger.info(f"Details fetched successfully for movie ID: {movie_id}")
        return details
    except Exception as e:
        logger.error(f"Error fetching TMDB details for movie ID {movie_id}: {e}")
        traceback.print_exc()
        return None

@cache.memoize(timeout=86400)  # Cache for 24 hours
def get_top_movies(n=10):
    """Get top rated movies from TMDB."""
    try:
        logger.info(f"Fetching top {n} movies from TMDB...")
        url = f"{TMDB_BASE_URL}/movie/top_rated"
        params = {
            "language": "en-US",
            "page": 1
        }
        
        response = requests.get(url, headers=TMDB_HEADERS, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Get top n movies
        movies = data.get("results", [])[:n]
        
        # Get complete details for each movie
        movie_details = []
        for movie in movies:
            details = fetch_movie_details(movie["id"])
            if details:
                movie_details.append(details)
        
        logger.info(f"Successfully fetched {len(movie_details)} top movies")
        return movie_details
    except Exception as e:
        logger.error(f"Error fetching top movies: {e}")
        traceback.print_exc()
        return []

@cache.memoize(timeout=86400)  # Cache for 24 hours
def get_popular_movies(n=10):
    """Get popular movies from TMDB."""
    try:
        logger.info(f"Fetching {n} popular movies from TMDB...")
        url = f"{TMDB_BASE_URL}/movie/popular"
        params = {
            "language": "en-US",
            "page": 1
        }
        
        response = requests.get(url, headers=TMDB_HEADERS, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Get top n movies
        movies = data.get("results", [])[:n]
        
        # Get complete details for each movie
        movie_details = []
        for movie in movies:
            details = fetch_movie_details(movie["id"])
            if details:
                movie_details.append(details)
        
        logger.info(f"Successfully fetched {len(movie_details)} popular movies")
        return movie_details
    except Exception as e:
        logger.error(f"Error fetching popular movies: {e}")
        traceback.print_exc()
        return []

@cache.memoize(timeout=86400)  # Cache for 24 hours
def get_now_playing_movies(n=10):
    """Get movies currently in theaters from TMDB."""
    try:
        logger.info(f"Fetching {n} now playing movies from TMDB...")
        url = f"{TMDB_BASE_URL}/movie/now_playing"
        params = {
            "language": "en-US",
            "page": 1
        }
        
        response = requests.get(url, headers=TMDB_HEADERS, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Get top n movies
        movies = data.get("results", [])[:n]
        
        # Get complete details for each movie
        movie_details = []
        for movie in movies:
            details = fetch_movie_details(movie["id"])
            if details:
                movie_details.append(details)
        
        logger.info(f"Successfully fetched {len(movie_details)} now playing movies")
        return movie_details
    except Exception as e:
        logger.error(f"Error fetching now playing movies: {e}")
        traceback.print_exc()
        return []

@cache.memoize(timeout=86400)  # Cache for 24 hours
def get_movies_by_genre(genre_id, n=10):
    """Get movies by genre from TMDB."""
    try:
        logger.info(f"Fetching {n} movies for genre ID {genre_id} from TMDB...")
        url = f"{TMDB_BASE_URL}/discover/movie"
        params = {
            "with_genres": genre_id,
            "sort_by": "popularity.desc",
            "language": "en-US",
            "page": 1
        }
        
        response = requests.get(url, headers=TMDB_HEADERS, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Get top n movies
        movies = data.get("results", [])[:n]
        
        # Get complete details for each movie
        movie_details = []
        for movie in movies:
            details = fetch_movie_details(movie["id"])
            if details:
                movie_details.append(details)
        
        logger.info(f"Successfully fetched {len(movie_details)} movies for genre ID {genre_id}")
        return movie_details
    except Exception as e:
        logger.error(f"Error fetching movies for genre ID {genre_id}: {e}")
        traceback.print_exc()
        return []

def recommend(movie_title, n=5):
    """Recommend movies based on the KNN model."""
    try:
        logger.info(f"Generating recommendations for: {movie_title}")
        
        if movie_title not in movie_list:
            logger.warning(f"Movie '{movie_title}' not found in dataset")
            return []

        # Log the movie we're finding recommendations for
        index = movies[movies['title'] == movie_title].index[0]
        logger.info(f"Found movie at index: {index}")
        
        try:
            # Make sure the KNN model is loaded and has _fit_X attribute
            if knn_model is None:
                logger.error("KNN model is not loaded")
                return []
                
            if not hasattr(knn_model, '_fit_X'):
                logger.error("KNN model doesn't have _fit_X attribute")
                return []
            
            # Log the shape of _fit_X to debug
            logger.debug(f"Shape of knn_model._fit_X: {knn_model._fit_X.shape}")
            
            # Access the feature vector and reshape properly
            feature_vector = knn_model._fit_X[index]
            if feature_vector.ndim == 1:
                feature_vector = feature_vector.reshape(1, -1)
                logger.debug(f"Reshaped feature vector to: {feature_vector.shape}")
            
            # Get nearest neighbors with distances
            distances, indices = knn_model.kneighbors(feature_vector, n_neighbors=n+1)
            logger.info(f"Found {len(indices.flatten())-1} neighbors")
            
            # Get recommendations excluding the input movie
            recommended_indices = indices.flatten()[1:]
            recommended_distances = distances.flatten()[1:]
            
            # Convert distances to similarity scores (1 = identical, 0 = completely different)
            # For cosine distance, we use 1 - distance as the similarity score
            similarity_scores = [round((1 - distance) * 100, 1) for distance in recommended_distances]
            
            logger.info(f"Recommended indices: {recommended_indices}")
            logger.info(f"Similarity scores: {similarity_scores}")
            
            # Get movie details for each recommendation
            recommendations = []
            for i, idx in enumerate(recommended_indices):
                title = movies.iloc[idx]['title']
                similarity = similarity_scores[i]
                logger.info(f"Processing recommendation: {title} (Similarity: {similarity}%)")
                
                # Search for the movie on TMDB
                search_results = search_movie(title)
                if search_results:
                    # Get the full details for the movie
                    movie_id = search_results[0]['id']
                    details = fetch_movie_details(movie_id)
                    if details:
                        # Add similarity score to the details
                        details['similarity'] = similarity
                        recommendations.append(details)
                        logger.info(f"Added recommendation: {title} with {similarity}% similarity")
                    else:
                        logger.warning(f"Failed to get details for {title}")
                else:
                    logger.warning(f"No search results found for {title}")
            
            logger.info(f"Returning {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in KNN recommendation: {e}")
            traceback.print_exc()
            return []
            
    except Exception as e:
        logger.error(f"Error in recommendation function: {e}")
        traceback.print_exc()
        return []

@app.route('/')
def home():
    """Home page with top-rated and popular movies."""
    top_movies = get_top_movies(10)
    popular_movies = get_popular_movies(10)
    now_playing = get_now_playing_movies(10)
    
    return render_template('index.html', 
                          top_movies=top_movies, 
                          popular_movies=popular_movies,
                          now_playing=now_playing)

@app.route('/movie/<int:movie_id>')
def movie_details_by_id(movie_id):
    """Display movie details by TMDB ID."""
    details = fetch_movie_details(movie_id)
    if not details:
        return render_template('error.html', message="Movie not found")
    return render_template('movie_details.html', movie=details)

@app.route('/movie/<path:movie_title>')
def movie_details_by_title(movie_title):
    """Search for a movie by title and redirect to its details page."""
    search_results = search_movie(movie_title)
    if not search_results:
        return render_template('error.html', message=f"Movie '{movie_title}' not found")
    
    movie_id = search_results[0]['id']
    return redirect(url_for('movie_details_by_id', movie_id=movie_id))

@app.route('/search')
def search_results():
    """Search for movies and display results."""
    query = request.args.get('query', '')
    if not query:
        return redirect(url_for('home'))
    
    search_results = search_movie(query)
    
    # Get full details for top search results
    movie_details = []
    for movie in search_results[:12]:  # Limit to top 12 results
        details = fetch_movie_details(movie['id'])
        if details:
            movie_details.append(details)
    
    return render_template('search_results.html', 
                           query=query, 
                           movies=movie_details,
                           result_count=len(search_results))

@app.route('/api/search')
def api_search():
    """API endpoint for movie search (used for autocomplete)."""
    query = request.args.get('q', '')
    if not query or len(query) < 2:
        return jsonify({"results": []})
    
    results = search_movie(query)
    simplified_results = [{
        "id": movie.get("id"),
        "title": movie.get("title"),
        "year": movie.get("release_date", "")[:4] if movie.get("release_date") else "",
        "poster": f"{TMDB_IMAGE_BASE_URL}w92{movie.get('poster_path')}" if movie.get("poster_path") else None
    } for movie in results[:10]]
    
    return jsonify({"results": simplified_results})

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """API endpoint for movie recommendations."""
    movie_id = request.form.get('movie_id')
    logger.info(f"Recommendation request for movie ID: '{movie_id}'")
    
    if not movie_id:
        logger.warning("No movie ID provided in request")
        return jsonify({'error': 'No movie ID provided'}), 400
    
    try:
        # Get movie details
        movie_details = fetch_movie_details(movie_id)
        if not movie_details:
            logger.warning(f"Movie with ID '{movie_id}' not found")
            return jsonify({'error': 'Movie not found'}), 404
        
        # Get recommendations from TMDB
        recommendations = movie_details.get('recommendations', [])
        
        if not recommendations:
            logger.warning(f"No recommendations found for movie ID '{movie_id}'")
            return jsonify({'error': 'No recommendations found'}), 404
        
        logger.info(f"Successfully returning {len(recommendations)} recommendations")
        return jsonify(recommendations)
    except Exception as e:
        logger.error(f"Error in recommendation endpoint: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/genres/<int:genre_id>')
def genre_movies(genre_id):
    """Display movies by genre."""
    # Map of genre IDs to names
    genres = {
        28: "Action",
        12: "Adventure",
        16: "Animation",
        35: "Comedy",
        80: "Crime",
        99: "Documentary",
        18: "Drama",
        10751: "Family",
        14: "Fantasy",
        36: "History",
        27: "Horror",
        10402: "Music",
        9648: "Mystery",
        10749: "Romance",
        878: "Science Fiction",
        10770: "TV Movie",
        53: "Thriller",
        10752: "War",
        37: "Western"
    }
    
    genre_name = genres.get(genre_id, "Unknown")
    movies = get_movies_by_genre(genre_id, 20)
    
    return render_template('genre.html', 
                          genre_name=genre_name,
                          genre_id=genre_id, 
                          movies=movies)

@app.route('/about')
def about():
    """About page."""
    return render_template('about.html')

@app.route('/contact')
def contact():
    """Contact page."""
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)