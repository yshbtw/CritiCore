# CineMate - Movie Recommendation Platform

CineMate is a modern, responsive web application that provides personalized movie recommendations based on user preferences. The platform uses machine learning to suggest movies similar to ones users have enjoyed in the past.

## Features

- **Personalized Movie Recommendations**: Utilizes a k-nearest neighbors (KNN) algorithm to suggest movies tailored to user preferences.
- **Extensive Movie Database**: Access to a vast collection of movies with detailed information sourced from The Movie Database (TMDB) API.
- **Responsive Design**: Optimized for all devices, from mobile phones to desktop computers.
- **Intuitive UI**: Clean, modern interface with high contrast for better accessibility.
- **Real-time Search**: Instantly search for movies with autocomplete functionality.
- **Movie Details**: Comprehensive information about each movie, including cast, crew, ratings, and more.
- **Genre Exploration**: Browse movies by genre.

## Tech Stack

- **Backend**: Python with Flask
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5
- **Data Processing**: Pandas for data manipulation
- **Machine Learning**: Scikit-learn for KNN model
- **API Integration**: TMDB API for movie data

## Installation and Setup

1. **Clone the Repository**
   ```
   git clone https://github.com/yshbtw/CritiCore.git
   cd CritiCore
   ```

2. **Install Dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**
   Create a `.env` file in the root directory with the following:
   ```
   TMDB_API_KEY=your_tmdb_api_key
   TMDB_ACCESS_TOKEN=your_tmdb_access_token
   ```

4. **Run the Application**
   ```
   python app.py
   ```

5. **Access the Website**
   Open your browser and navigate to `http://localhost:5000`

## Project Structure

- `app.py`: Main Flask application file
- `requirements.txt`: Python dependencies
- `templates/`: HTML templates
  - `index.html`: Homepage
  - `movie_details.html`: Individual movie page
  - `search_results.html`: Search results page
  - `genre.html`: Genre-specific movie listings
  - `about.html`: About page
  - `contact.html`: Contact information and form
  - `error.html`: Error page
- `knn_model.pkl`: Pre-trained KNN model for recommendations
- `Movie_list.pkl`: Processed movie dataset

## API Usage

The application uses The Movie Database (TMDB) API to fetch movie data. Key endpoints include:

- Movie search
- Movie details retrieval
- Popular movies listings
- Top-rated movies listings
- Now playing movies
- Genre-specific movie listings

## Accessibility Features

- High contrast color scheme
- Responsive design for all devices
- Accessible form elements with proper labeling
- Keyboard navigation support
- Internal page navigation links

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [The Movie Database (TMDB)](https://www.themoviedb.org/) for providing the movie data API
- [Bootstrap](https://getbootstrap.com/) for the responsive front-end framework
- [Font Awesome](https://fontawesome.com/) for the icons
- [Google Fonts](https://fonts.google.com/) for the typography 