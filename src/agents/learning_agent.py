import os
import json
import pickle
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)    

class LearningAgent:
    """
    Learning agent that improves search results through user feedback and interactions.
    Implements continuous learning approaches and integrates with RAG-Gym for 
    state-of-the-art retrieval augmented generation.
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
        
        # RAG-Gym integration components
        self.query_rewriters = {}  # {pattern: rewrite_function}
        self.content_retrievers = {}  # {strategy_name: retriever_function} 
        self.result_rerankers = {}  # {model_name: reranker_function}
        
        # Load existing data if available
        self._load_data()
        self._initialize_rag_components()
        
        logger.info("Learning agent initialized with RAG-Gym integration")

    def _initialize_rag_components(self):
        """Initialize RAG-Gym integration components"""
        # Register query rewriters
        self.query_rewriters = {
            "domain_enhance": self._domain_enhance_rewriter,
            "historical": self._historical_query_rewriter,
            "personalized": self._personalized_query_rewriter
        }
        
        # Register content retrievers (strategies for finding content)
        self.content_retrievers = {
            "bm25": lambda q, n: self.retrieval_agent._bm25_search(q, n),
            "vector": lambda q, n: self.retrieval_agent._faiss_search(q, n),
            "hybrid": self._hybrid_retrieval
        }
        
        # Register rerankers
        self.result_rerankers = {
            "personalized": self._personalized_reranker,
            "relevance": self._relevance_reranker,
            "diversity": self._diversity_reranker
        }
        
        logger.info("RAG-Gym components initialized")
    
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
            
        if hasattr(paper, 'doi') and paper.doi:
            return f"doi:{paper.doi}"
            
        # Fallback to title-based ID
        return f"title:{paper.title.lower().strip()[:50]}"

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
        
        # Update source preferences if available
        for paper in clicked_papers:
            if hasattr(paper, 'source') and paper.source:
                self.user_profiles[user_id][f"source:{paper.source}"] += 1.0
        
        # Apply decay to old preferences (time-based forgetting)
        decay_factor = 0.95
        for topic in self.user_profiles[user_id]:
            self.user_profiles[user_id][topic] *= decay_factor
        
        # Normalize weights
        self._normalize_user_profile(user_id)
    
    def _normalize_user_profile(self, user_id):
        """Normalize user profile weights to prevent unbounded growth"""
        total = sum(self.user_profiles[user_id].values())
        if total > 0:
            for topic in self.user_profiles[user_id]:
                self.user_profiles[user_id][topic] /= total

    def _extract_topics(self, text):
        """Extract topics from text - simplified implementation"""
        if not text:
            return []
            
        # In a real implementation, you would use NLP techniques
        # This is a simplified approach
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
            list: Reranked results based on user preferences
        """
        if not self.user_profiles.get(user_id) or not base_results:
            return base_results
        
        profile = self.user_profiles[user_id]
        scored_results = []
        
        # Score each paper based on user preferences
        for paper in base_results:
            # Base relevance score (1.0)
            score = 1.0
            
            # Add topic preference score
            paper_topics = set(self._extract_topics(paper.title) + self._extract_topics(paper.abstract))
            topic_score = sum(profile.get(topic, 0) for topic in paper_topics)
            
            # Add source preference score
            source = getattr(paper, 'source', None)
            source_score = profile.get(f"source:{source}", 0) if source else 0
            
            # Combine scores (weights could be tuned)
            final_score = score + (0.5 * topic_score) + (0.3 * source_score)
            
            scored_results.append((paper, final_score))
        
        # Sort by score in descending order
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return reranked papers
        return [paper for paper, score in scored_results]

    def _calculate_source_preference(self, user_id, source):
        """Calculate user's preference for a particular source"""
        if not user_id or not source:
            return 0.0
            
        profile = self.user_profiles.get(user_id, {})
        return profile.get(f"source:{source}", 0.0)

    def recommend_related_papers(self, paper, limit=5):
        """
        Recommend papers related to the given paper
        
        Args:
            paper: Paper object to find related papers for
            limit (int): Maximum number of recommendations to return
            
        Returns:
            list: List of related Paper objects
        """
        paper_id = self._get_paper_id(paper)
        
        # Check cache first
        if paper_id in self.similar_papers_cache:
            return self.similar_papers_cache[paper_id][:limit]
        
        # Extract topics from the paper
        paper_topics = set(self._extract_topics(paper.title) + self._extract_topics(paper.abstract))
        
        # If no topics found, return empty list
        if not paper_topics:
            return []
        
        # Get all papers from retrieval agent
        all_papers = self.retrieval_agent.papers
        
        # Score each paper by topic similarity
        scored_papers = []
        for other_paper in all_papers:
            if self._get_paper_id(other_paper) == paper_id:
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

    # RAG-GYM INTEGRATION METHODS
    
    # Query Rewriters
    def _domain_enhance_rewriter(self, query: str, user_id: str = None) -> str:
        """Enhances query with domain-specific terminology"""
        return self.query_agent._enhance_domain_query(query)
    
    def _historical_query_rewriter(self, query: str, user_id: str = None) -> str:
        """Uses past successful queries to improve the current query"""
        return self.improve_query(query)
    
    def _personalized_query_rewriter(self, query: str, user_id: str) -> str:
        """Enhances query using user profile information"""
        if not user_id or user_id not in self.user_profiles:
            return query
            
        # Get top user interests
        profile = self.user_profiles[user_id]
        top_interests = sorted(
            [(topic, weight) for topic, weight in profile.items() 
             if not topic.startswith("source:")],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        # Add top interests to query if relevant
        if top_interests:
            interest_terms = [term for term, _ in top_interests]
            # Only add terms that aren't already in the query
            query_terms = query.lower().split()
            new_terms = [term for term in interest_terms 
                        if term.lower() not in query_terms]
            
            if new_terms:
                enhanced_query = f"{query} {' '.join(new_terms)}"
                logger.info(f"Personalized query enhancement: '{query}' -> '{enhanced_query}'")
                return enhanced_query
                
        return query
    
    # Content Retrievers
    def _hybrid_retrieval(self, query: str, top_k: int = 10) -> List[Any]:
        """Combines multiple retrieval methods for better results"""
        # Get results from both BM25 and vector search
        bm25_results = self.retrieval_agent._bm25_search(query, top_k)
        vector_results = self.retrieval_agent._faiss_search(query, top_k)
        
        # Combine results with rank fusion
        all_results = {}
        
        # Add BM25 results with rank
        for i, paper in enumerate(bm25_results):
            paper_id = self._get_paper_id(paper)
            all_results[paper_id] = {
                'paper': paper,
                'bm25_rank': i + 1,
                'vector_rank': top_k + 2  # Default low rank if not in vector results
            }
        
        # Add/update vector results with rank
        for i, paper in enumerate(vector_results):
            paper_id = self._get_paper_id(paper)
            if paper_id in all_results:
                all_results[paper_id]['vector_rank'] = i + 1
            else:
                all_results[paper_id] = {
                    'paper': paper,
                    'bm25_rank': top_k + 2,  # Default low rank if not in BM25 results
                    'vector_rank': i + 1
                }
        
        # Calculate combined score (reciprocal rank fusion)
        k = 60  # Constant to smooth rankings
        for paper_id in all_results:
            all_results[paper_id]['score'] = 1/(k + all_results[paper_id]['bm25_rank']) + 1/(k + all_results[paper_id]['vector_rank'])
        
        # Sort by combined score and return papers
        sorted_results = sorted(all_results.values(), key=lambda x: x['score'], reverse=True)
        return [item['paper'] for item in sorted_results[:top_k]]
    
    # Result Rerankers
    def _personalized_reranker(self, papers: List[Any], user_id: str, query: str) -> List[Any]:
        """Reranks results based on user preferences"""
        return self.get_personalized_results(user_id, query, papers)
    
    def _relevance_reranker(self, papers: List[Any], user_id: str, query: str) -> List[Any]:
        """Reranks results based on relevance to query"""
        if not papers:
            return []
            
        query_terms = self._extract_topics(query)
        if not query_terms:
            return papers
            
        # Score papers by relevance to query
        scored_papers = []
        
        for paper in papers:
            title_terms = self._extract_topics(paper.title)
            abstract_terms = self._extract_topics(paper.abstract)
            
            # Count matches in title (weighted higher) and abstract
            title_matches = sum(1 for term in query_terms if term in title_terms)
            abstract_matches = sum(1 for term in query_terms if term in abstract_terms)
            
            # Calculate relevance score
            relevance_score = (3 * title_matches) + abstract_matches
            scored_papers.append((paper, relevance_score))
        
        # Sort by relevance score
        scored_papers.sort(key=lambda x: x[1], reverse=True)
        return [paper for paper, _ in scored_papers]
    
    def _diversity_reranker(self, papers: List[Any], user_id: str, query: str) -> List[Any]:
        """Reranks results to increase diversity"""
        if len(papers) <= 5:
            return papers
            
        # Extract all topics
        all_topics = set()
        paper_topics = {}
        
        for paper in papers:
            topics = set(self._extract_topics(paper.title) + self._extract_topics(paper.abstract))
            paper_topics[self._get_paper_id(paper)] = topics
            all_topics.update(topics)
        
        # Initialize with highest relevance paper
        # For simplicity, assume the first paper is most relevant
        selected = [papers[0]]
        remaining = papers[1:]
        
        # Keep track of covered topics
        covered_topics = paper_topics[self._get_paper_id(papers[0])].copy()
        
        # Greedily select papers that add most new topics
        while remaining and len(selected) < len(papers):
            best_paper = None
            best_new_topics = -1
            
            for paper in remaining:
                paper_id = self._get_paper_id(paper)
                new_topics = len(paper_topics[paper_id] - covered_topics)
                
                if new_topics > best_new_topics:
                    best_new_topics = new_topics
                    best_paper = paper
            
            if best_paper:
                selected.append(best_paper)
                remaining.remove(best_paper)
                covered_topics.update(paper_topics[self._get_paper_id(best_paper)])
            else:
                # If no paper adds new topics, add the first remaining one
                selected.append(remaining[0])
                remaining = remaining[1:]
        
        return selected
    
    def get_rag_components(self):
        """
        Get all RAG components for external integration
        
        Returns:
            dict: Dictionary of rewriters, retrievers, and rerankers
        """
        return {
            'rewriters': self.query_rewriters,
            'retrievers': self.content_retrievers,
            'rerankers': self.result_rerankers
        }
    
    def rag_pipeline(self, query: str, user_id: str = "anonymous", retriever: str = "hybrid", 
                     rewriter: str = "domain_enhance", reranker: str = "personalized", 
                     top_k: int = 10) -> Tuple[List[Any], str]:
        """
        Run a complete RAG pipeline using specified components
        
        Args:
            query: User query
            user_id: User identifier
            retriever: Name of retrieval method to use
            rewriter: Name of query rewriter to use
            reranker: Name of reranker to use
            top_k: Number of results to return
            
        Returns:
            tuple: (results, rewritten_query)
        """
        # Step 1: Rewrite query
        if rewriter in self.query_rewriters:
            rewritten_query = self.query_rewriters[rewriter](query, user_id)
        else:
            rewritten_query = query
            logger.warning(f"Rewriter '{rewriter}' not found, using original query")
        
        # Step 2: Retrieve content
        if retriever in self.content_retrievers:
            retrieved_papers = self.content_retrievers[retriever](rewritten_query, top_k)
        else:
            retrieved_papers = self.retrieval_agent.search(rewritten_query)
            logger.warning(f"Retriever '{retriever}' not found, using default search")
        
        # Step 3: Rerank results
        if reranker in self.result_rerankers and retrieved_papers:
            final_papers = self.result_rerankers[reranker](retrieved_papers, user_id, query)
        else:
            final_papers = retrieved_papers
            if retrieved_papers:
                logger.warning(f"Reranker '{reranker}' not found, using retrieved results directly")
        
        return final_papers, rewritten_query
        
    def train_recommendation_model(self):
        """Train a recommendation model based on collected data"""
        # In a production system, this would train a more sophisticated model
        # like a matrix factorization or neural network
        
        # For now, just ensure our data is saved
        self._save_data()
        logger.info("Recommendation model training complete")