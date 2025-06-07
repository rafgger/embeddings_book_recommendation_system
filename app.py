from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class BookRecommendationSystem:
    def __init__(self, csv_path: str = 'books.csv'):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.books_df = None
        self.embeddings = None
        self.embeddings_file = 'book_embeddings.pkl'
        self.load_data(csv_path)
        
    def load_data(self, csv_path: str):
        """Load and preprocess the books dataset"""
        try:
            # Read the CSV file
            self.books_df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(self.books_df)} books from {csv_path}")
            
            # Clean and preprocess the data
            self.books_df = self.books_df.dropna(subset=['Book-Title', 'Book-Author'])
            
            # Create combined text for embeddings (title + author + year)
            self.books_df['combined_text'] = (
                self.books_df['Book-Title'].astype(str) + ' by ' + 
                self.books_df['Book-Author'].astype(str) + ' (' + 
                self.books_df['Year-Of-Publication'].astype(str) + ')'
            )
            
            # Load or create embeddings
            self.load_or_create_embeddings()
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            # Create sample data for demonstration
            self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample data based on the provided CSV snippet"""
        sample_books = [
            {'ISBN': '0195153448', 'Book-Title': 'Classical Mythology', 'Book-Author': 'Mark P. O. Morford', 'Year-Of-Publication': 2002, 'Publisher': 'Oxford University Press'},
            {'ISBN': '0002005018', 'Book-Title': 'Clara Callan', 'Book-Author': 'Richard Bruce Wright', 'Year-Of-Publication': 2001, 'Publisher': 'HarperFlamingo Canada'},
            {'ISBN': '0060973129', 'Book-Title': 'Decision in Normandy', 'Book-Author': 'Carlo D\'Este', 'Year-Of-Publication': 1991, 'Publisher': 'HarperPerennial'},
            {'ISBN': '0374157065', 'Book-Title': 'Flu: The Story of the Great Influenza Pandemic of 1918', 'Book-Author': 'Gina Bari Kolata', 'Year-Of-Publication': 1999, 'Publisher': 'Farrar Straus Giroux'},
            {'ISBN': '0393045218', 'Book-Title': 'The Mummies of Urumchi', 'Book-Author': 'E. J. W. Barber', 'Year-Of-Publication': 1999, 'Publisher': 'W. W. Norton & Company'},
            {'ISBN': '0399135782', 'Book-Title': 'The Kitchen God\'s Wife', 'Book-Author': 'Amy Tan', 'Year-Of-Publication': 1991, 'Publisher': 'Putnam Pub Group'},
            {'ISBN': '0425176428', 'Book-Title': 'What If?: The World\'s Foremost Military Historians Imagine What Might Have Been', 'Book-Author': 'Robert Cowley', 'Year-Of-Publication': 2000, 'Publisher': 'Berkley Publishing Group'},
            {'ISBN': '0671870432', 'Book-Title': 'PLEADING GUILTY', 'Book-Author': 'Scott Turow', 'Year-Of-Publication': 1993, 'Publisher': 'Audioworks'},
            {'ISBN': '0679425608', 'Book-Title': 'Under the Black Flag: The Romance and the Reality of Life Among the Pirates', 'Book-Author': 'David Cordingly', 'Year-Of-Publication': 1996, 'Publisher': 'Random House'},
            {'ISBN': '074322678X', 'Book-Title': 'Where You\'ll Find Me: And Other Stories', 'Book-Author': 'Ann Beattie', 'Year-Of-Publication': 2002, 'Publisher': 'Scribner'},
            {'ISBN': '0440234743', 'Book-Title': 'The Testament', 'Book-Author': 'John Grisham', 'Year-Of-Publication': 1999, 'Publisher': 'Dell'},
            {'ISBN': '0452264464', 'Book-Title': 'Beloved', 'Book-Author': 'Toni Morrison', 'Year-Of-Publication': 1994, 'Publisher': 'Plume'},
            {'ISBN': '0345402871', 'Book-Title': 'Airframe', 'Book-Author': 'Michael Crichton', 'Year-Of-Publication': 1997, 'Publisher': 'Ballantine Books'},
            {'ISBN': '0345417623', 'Book-Title': 'Timeline', 'Book-Author': 'MICHAEL CRICHTON', 'Year-Of-Publication': 2000, 'Publisher': 'Ballantine Books'},
            {'ISBN': '0684823802', 'Book-Title': 'OUT OF THE SILENT PLANET', 'Book-Author': 'C.S. Lewis', 'Year-Of-Publication': 1996, 'Publisher': 'Scribner'},
            {'ISBN': '0375759778', 'Book-Title': 'Prague : A Novel', 'Book-Author': 'ARTHUR PHILLIPS', 'Year-Of-Publication': 2003, 'Publisher': 'Random House Trade Paperbacks'},
            # Add some fantasy books for better Lord of the Rings recommendations
            {'ISBN': '0547928270', 'Book-Title': 'The Hobbit', 'Book-Author': 'J.R.R. Tolkien', 'Year-Of-Publication': 1937, 'Publisher': 'Houghton Mifflin Harcourt'},
            {'ISBN': '0345339711', 'Book-Title': 'The Fellowship of the Ring', 'Book-Author': 'J.R.R. Tolkien', 'Year-Of-Publication': 1954, 'Publisher': 'Ballantine Books'},
            {'ISBN': '0345339738', 'Book-Title': 'The Two Towers', 'Book-Author': 'J.R.R. Tolkien', 'Year-Of-Publication': 1954, 'Publisher': 'Ballantine Books'},
            {'ISBN': '0345339746', 'Book-Title': 'The Return of the King', 'Book-Author': 'J.R.R. Tolkien', 'Year-Of-Publication': 1955, 'Publisher': 'Ballantine Books'},
            {'ISBN': '0441013597', 'Book-Title': 'Dune', 'Book-Author': 'Frank Herbert', 'Year-Of-Publication': 1965, 'Publisher': 'Ace Books'},
            {'ISBN': '0553293354', 'Book-Title': 'Foundation', 'Book-Author': 'Isaac Asimov', 'Year-Of-Publication': 1951, 'Publisher': 'Bantam Spectra'},
            {'ISBN': '0345404475', 'Book-Title': 'A Game of Thrones', 'Book-Author': 'George R.R. Martin', 'Year-Of-Publication': 1996, 'Publisher': 'Bantam Spectra'},
            {'ISBN': '0345535529', 'Book-Title': 'The Name of the Wind', 'Book-Author': 'Patrick Rothfuss', 'Year-Of-Publication': 2007, 'Publisher': 'DAW Books'},
            {'ISBN': '0765311178', 'Book-Title': 'The Way of Kings', 'Book-Author': 'Brandon Sanderson', 'Year-Of-Publication': 2010, 'Publisher': 'Tor Books'},
        ]
        
        self.books_df = pd.DataFrame(sample_books)
        self.books_df['combined_text'] = (
            self.books_df['Book-Title'].astype(str) + ' by ' + 
            self.books_df['Book-Author'].astype(str) + ' (' + 
            self.books_df['Year-Of-Publication'].astype(str) + ')'
        )
        self.load_or_create_embeddings()
    
    def load_or_create_embeddings(self):
        """Load existing embeddings or create new ones"""
        if os.path.exists(self.embeddings_file):
            try:
                with open(self.embeddings_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
                logger.info("Loaded existing embeddings")
                return
            except Exception as e:
                logger.warning(f"Error loading embeddings: {str(e)}")
        
        # Create new embeddings
        logger.info("Creating new embeddings...")
        texts = self.books_df['combined_text'].tolist()
        self.embeddings = self.model.encode(texts)
        
        # Save embeddings
        try:
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.embeddings, f)
            logger.info("Saved embeddings to file")
        except Exception as e:
            logger.warning(f"Error saving embeddings: {str(e)}")
    
    def find_similar_books(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find books similar to the query using embeddings"""
        try:
            # Encode the query
            query_embedding = self.model.encode([query])
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Get top k similar books
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            recommendations = []
            for idx in top_indices:
                book = self.books_df.iloc[idx]
                recommendations.append({
                    'title': book['Book-Title'],
                    'author': book['Book-Author'],
                    'year': book['Year-Of-Publication'],
                    'publisher': book['Publisher'],
                    'similarity': float(similarities[idx])
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error finding similar books: {str(e)}")
            return []
    
    def search_book_in_dataset(self, query: str) -> List[Dict[str, Any]]:
        """Search for books in the dataset that match the query"""
        query_lower = query.lower()
        matches = []
        
        for idx, row in self.books_df.iterrows():
            title_lower = str(row['Book-Title']).lower()
            author_lower = str(row['Book-Author']).lower()
            
            if query_lower in title_lower or query_lower in author_lower:
                matches.append({
                    'title': row['Book-Title'],
                    'author': row['Book-Author'],
                    'year': row['Year-Of-Publication'],
                    'publisher': row['Publisher']
                })
        
        return matches

# Initialize the recommendation system
recommender = BookRecommendationSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        favorite_book = data.get('book', '').strip()
        
        if not favorite_book:
            return jsonify({'error': 'Please provide a book title'}), 400
        
        # Find recommendations
        recommendations = recommender.find_similar_books(favorite_book, top_k=10)
        
        # Also search for exact matches in dataset
        exact_matches = recommender.search_book_in_dataset(favorite_book)
        
        return jsonify({
            'query': favorite_book,
            'recommendations': recommendations,
            'exact_matches': exact_matches
        })
        
    except Exception as e:
        logger.error(f"Error in recommend endpoint: {str(e)}")
        return jsonify({'error': 'An error occurred while processing your request'}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'books_count': len(recommender.books_df)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)