import os
import json
import pickle
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class LearningAgent:
    """
    Learning agent that improves search results through user feedback and interactions.
    Implements continuous learning and reinforcement learning approaches.
    """
    
    def __init__(self, retrieval_agent, query_agent, data_dir=None):
        """
        Initialize the LearningAgent with references to other agents.
        
        Args:
            retrieval_agent: The RetrievalAgent instance
            query_agent: The QueryAgent instance
            data_dir: Directory to store learning data
        """
        self.retrieval_agent = retrieval_agent
        self.query_agent = query_agent
        
        # Set up data storage
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data" / "learning"
        else:
            data_dir = Path(data_dir)
        
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Paths for saved data
        self.user_interactions_path = self.data_dir / "user_interactions.pkl"
        self.paper_embeddings_path = self.data_dir / "paper_embeddings.pkl"
        self.user_profiles_path = self.data_dir / "user_profiles.pkl"
        self.recommendation_model_path = self.data_dir / "recommendation_model.pkl"
        
        # Learning state
        self.user_interactions = defaultdict(list)  # {user_id: [{query, papers_shown, clicked, time, feedback}]}
        self.query_success_metrics = defaultdict(list)  # {query: [success_scores]}
        self.popular_papers = Counter()  # {paper_id: click_count}
        self.similar_papers_cache = {}  # {paper_id: [similar_paper_ids]}
        self.user_profiles = defaultdict(lambda: defaultdict(float))  # {user_id: {topic: weight}}
        
        # Load existing data if available
        self._load_data()
        
        logger.info("Learning agent initialized")
    
    def _load_data(self):
        """Load previously saved learning data"""
        try:
            if self.user_interactions_path.exists():
                with open(self.user_interactions_path, 'rb') as f:
                    self.user_interactions = pickle.load(f)
                logger.info(f"Loaded {sum(len(v) for v in self.user_interactions.values())} user interactions")
            
            if self.user_profiles_path.exists():
                with open(self.user_profiles_path, 'rb') as f:
                    self.user_profiles = pickle.load(f)
                logger.info(f"Loaded {len(self.user_profiles)} user profiles")
                
            # Extract popular papers from interactions
            for user_id, interactions in self.user_interactions.items():
                for interaction in interactions:
                    for paper_id in interaction.get('clicked', []):
                        self.popular_papers[paper_id] += 1
                        
        except Exception as e:
            logger.error(f"Error loading learning data: {str(e)}")
    
    def _save_data(self):
        """Save learning data to disk"""
        try:
            with open(self.user_interactions_path, 'wb') as f:
                pickle.dump(self.user_interactions, f)
                
            with open(self.user_profiles_path, 'wb') as f:
                pickle.dump(self.user_profiles, f)
                
            logger.info("Learning data saved successfully")
        except Exception as e:
            logger.error(f"Error saving learning data: {str(e)}")
    
    def record_user_interaction(self, user_id, query, results, clicked_papers, time_spent, explicit_feedback=None):
        """
        Record user search interaction for learning
        
        Args:
            user_id (str): User identifier
            query (str): Search query
            results (list): Papers shown to user
            clicked_papers (list): Papers user clicked on
            time_spent (float): Time spent viewing results (seconds)
            explicit_feedback (dict, optional): User ratings of papers {paper_id: rating}
        """
        # Get paper IDs for tracking
        result_ids = [self._get_paper_id(paper) for paper in results]
        clicked_ids = [self._get_paper_id(paper) for paper in clicked_papers]
        
        # Record interaction
        self.user_interactions[user_id].append({
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'papers_shown': result_ids,
            'clicked': clicked_ids,
            'time_spent': time_spent,
            'feedback': explicit_feedback
        })
        
        # Update popular papers counter
        for paper_id in clicked_ids:
            self.popular_papers[paper_id] += 1
        
        # Calculate implicit feedback score
        if results:
            success_score = len(clicked_papers) / len(results)
            self.query_success_metrics[query].append(success_score)
            
            # If query was particularly successful, learn from it
            if success_score > 0.5:
                self._improve_from_successful_query(query, clicked_papers, results)
        
        # Update user profile
        self._update_user_profile(user_id, query, clicked_papers)
        
        # Save data periodically (could be optimized to save less frequently)
        if len(self.user_interactions[user_id]) % 10 == 0:
            self._save_data()
        
        logger.info(f"Recorded interaction for user {user_id}: query='{query}', clicked={len(clicked_ids)}/{len(result_ids)}")
    
    def _get_paper_id(self, paper):
        """Generate a unique identifier for a paper"""
        if hasattr(paper, 'id') and paper.id:
            return paper.id
        
        # Create identifier from metadata if no ID exists
        if hasattr(paper, 'doi') and paper.doi:
            return f"doi:{paper.doi}"
        
        # Fall back to title-based ID
        return f"title:{paper.title}"
    
    def _update_user_profile(self, user_id, query, clicked_papers):
        """Update user profile based on interaction"""
        # Extract topics from query and papers
        query_topics = self._extract_topics(query)
        paper_topics = []
        for paper in clicked_papers:
            paper_topics.extend(self._extract_topics(paper.title))
            paper_topics.extend(self._extract_topics(paper.abstract))
        
        # Update profile with topics from this interaction
        for topic in query_topics:
            self.user_profiles[user_id][topic] += 0.5
        
        for topic in paper_topics:
            self.user_profiles[user_id][topic] += 1.0
        
        # Normalize weights
        total = sum(self.user_profiles[user_id].values())
        if total > 0:
            for topic in self.user_profiles[user_id]:
                self.user_profiles[user_id][topic] /= total
    
    def _extract_topics(self, text):
        """Extract topics from text - simplified implementation"""
        if not text:
            return []
            
        # In a real implementation, you would use NLP techniques
        # This is a very simplified approach
        words = text.lower().split()
        # Filter to keep only meaningful words (longer than 3 chars)
        return [word for word in words if len(word) > 3]
    
    def _improve_from_successful_query(self, query, clicked_papers, all_results):
        """Learn from successful queries"""
        # Extract features from successful results for reinforcement learning
        good_features = self._extract_features_from_papers(clicked_papers)
        
        # Get papers that weren't clicked
        unclicked_papers = [p for p in all_results if p not in clicked_papers]
        
        # Use this information to train retrieval_agent
        self.retrieval_agent.learn_from_feedback(query, clicked_papers, unclicked_papers)
    
    def _extract_features_from_papers(self, papers):
        """Extract features from papers for learning"""
        features = defaultdict(float)
        
        for paper in papers:
            # Add features based on paper metadata
            if hasattr(paper, 'publication_year') and paper.publication_year:
                features[f"year:{paper.publication_year}"] += 1
            
            if hasattr(paper, 'source') and paper.source:
                features[f"source:{paper.source}"] += 1
            
            # Add topic features
            for topic in self._extract_topics(paper.title):
                features[f"title_topic:{topic}"] += 1
                
            for topic in self._extract_topics(paper.abstract):
                features[f"abstract_topic:{topic}"] += 0.5
        
        # Normalize
        total = sum(features.values())
        if total > 0:
            for feature in features:
                features[feature] /= total
                
        return dict(features)
    
    def get_personalized_results(self, user_id, query, base_results):
        """
        Personalize search results for a specific user
        
        Args:
            user_id (str): User identifier
            query (str): Search query
            base_results (list): Initial search results
            
        Returns:
            list: Reranked search results
        """
        if not self.user_profiles.get(user_id) or not base_results:
            return base_results
        
        # Score each paper based on user profile
        scored_results = []
        for paper in base_results:
            # Base score from search
            score = 1.0
            
            # Topic matching score
            paper_topics = set(self._extract_topics(paper.title) + self._extract_topics(paper.abstract))
            
            profile = self.user_profiles[user_id]
            topic_score = sum(profile.get(topic, 0) for topic in paper_topics)
            
            # Source preference score
            source = getattr(paper, 'source', None)
            source_score = self._calculate_source_preference(user_id, source)
            
            # Combine scores (weights could be tuned)
            final_score = score + (0.5 * topic_score) + (0.3 * source_score)
            
            scored_results.append((paper, final_score))
        
        # Sort by score in descending order
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return reranked papers
        return [paper for paper, score in scored_results]
    
    def _calculate_source_preference(self, user_id, source):
        """Calculate user's preference for a particular source"""
        if not source or user_id not in self.user_interactions:
            return 0
            
        # Count how many papers user clicked from each source
        source_clicks = Counter()
        total_clicks = 0
        
        for interaction in self.user_interactions[user_id]:
            clicked_papers = interaction.get('clicked', [])
            for paper_id in clicked_papers:
                # In a real implementation, you'd need to retrieve the paper source
                # This is simplified
                if paper_id.startswith(f"source:{source}"):
                    source_clicks[source] += 1
                total_clicks += 1
        
        # Return normalized preference
        if total_clicks > 0:
            return source_clicks.get(source, 0) / total_clicks
        return 0
    
    def recommend_related_papers(self, paper, limit=5):
        """
        Recommend papers related to a specific paper
        
        Args:
            paper: Paper to find related papers for
            limit (int): Maximum number of recommendations
            
        Returns:
            list: Related papers
        """
        paper_id = self._get_paper_id(paper)
        
        # Check cache first
        if paper_id in self.similar_papers_cache:
            return self.similar_papers_cache[paper_id][:limit]
        
        # Find papers with similar topics
        paper_topics = set(self._extract_topics(paper.title) + self._extract_topics(paper.abstract))
        
        # In a real implementation, you'd use embeddings and similarity search
        # This is a simplified approach looking at all papers the agent knows about
        all_papers = self.retrieval_agent.get_all_papers()
        
        scored_papers = []
        for other_paper in all_papers:
            if other_paper.title == paper.title:
                continue  # Skip the same paper
                
            other_topics = set(self._extract_topics(other_paper.title) + self._extract_topics(other_paper.abstract))
            
            # Jaccard similarity between topics
            if not other_topics or not paper_topics:
                continue
                
            similarity = len(paper_topics.intersection(other_topics)) / len(paper_topics.union(other_topics))
            scored_papers.append((other_paper, similarity))
        
        # Sort by similarity
        scored_papers.sort(key=lambda x: x[1], reverse=True)
        
        # Cache results
        results = [p for p, score in scored_papers[:20]]  # Cache more than needed
        self.similar_papers_cache[paper_id] = results
        
        return results[:limit]
    
    def improve_query(self, query):
        """
        Improve query based on past learning
        
        Args:
            query (str): Original user query
            
        Returns:
            str: Improved query
        """
        # Simplistic query expansion based on similar successful queries
        query_terms = set(query.lower().split())
        expanded_terms = set(query_terms)
        
        # Look for similar successful queries
        for past_query, scores in self.query_success_metrics.items():
            if not scores:
                continue
                
            avg_score = sum(scores) / len(scores)
            if avg_score < 0.5:
                continue  # Skip unsuccessful queries
                
            past_terms = set(past_query.lower().split())
            
            # If queries are similar but not identical
            common_terms = query_terms.intersection(past_terms)
            if common_terms and len(common_terms) / len(query_terms) > 0.5:
                # Add terms from successful query
                expanded_terms.update(past_terms)
        
        # If we've significantly expanded the query
        if len(expanded_terms) > len(query_terms) * 1.5:
            expanded_query = " ".join(expanded_terms)
            logger.info(f"Expanded query '{query}' to '{expanded_query}'")
            return expanded_query
        
        return query
    
    def train_recommendation_model(self):
        """Train a recommendation model based on collected data"""
        # In a production system, this would train a more sophisticated model
        # like a matrix factorization or neural network
        
        # For now, just ensure our data is saved
        self._save_data()
        logger.info("Recommendation model training complete")