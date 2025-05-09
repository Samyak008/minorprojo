"""Shared text processing utilities for all agents"""
import re
from collections import Counter
import nltk

# Download NLTK resources if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Initialize stopwords once
STOPWORDS = set(stopwords.words('english'))

def tokenize_and_normalize(text):
    """Tokenize and normalize text for processing"""
    if not text:
        return []
        
    # Convert to lowercase and tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords, punctuation, and short tokens
    tokens = [
        token for token in tokens
        if token not in STOPWORDS
        and re.match(r'^[a-z]+$', token)
        and len(token) > 2
    ]
    
    return tokens

def extract_topics(text, max_topics=10):
    """Extract main topics from text based on term frequency"""
    if not text:
        return []
        
    tokens = tokenize_and_normalize(text)
    
    # Count term frequencies
    term_counts = Counter(tokens)
    
    # Return most common terms as topics
    return [term for term, _ in term_counts.most_common(max_topics)]

def calculate_text_similarity(text1, text2):
    """Calculate similarity between two text strings"""
    if not text1 or not text2:
        return 0.0
        
    tokens1 = set(tokenize_and_normalize(text1))
    tokens2 = set(tokenize_and_normalize(text2))
    
    # Calculate Jaccard similarity
    if not tokens1 or not tokens2:
        return 0.0
        
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    
    return len(intersection) / len(union)