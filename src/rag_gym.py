import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path
import os
import json
import pickle
from collections import defaultdict
from datetime import datetime
import time
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class RAGGym:
    """
    RAG-Gym implementation for process-driven retrieval, fine-tuned query optimization,
    and reinforcement learning-driven personalization.
    """
    
    def __init__(self, retrieval_agent=None, query_agent=None, learning_agent=None, data_dir=None):
        """
        Initialize RAG-Gym with references to system agents
        
        Args:
            retrieval_agent: The system's RetrievalAgent
            query_agent: The system's QueryAgent
            learning_agent: The system's LearningAgent
            data_dir: Directory to store RAG-Gym data
        """
        self.retrieval_agent = retrieval_agent
        self.query_agent = query_agent
        self.learning_agent = learning_agent
        
        # Set up data directory
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data" / "rag_gym"
        else:
            data_dir = Path(data_dir)
        
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
        # File paths for models and data
        self.sft_model_path = self.data_dir / "sft_model.pkl"
        self.prm_model_path = self.data_dir / "prm_model.pkl"
        self.process_trajectories_path = self.data_dir / "process_trajectories.json"
        self.reward_model_path = self.data_dir / "reward_model.pkl"
        
        # RAG-Gym state
        self.process_trajectories = []
        self.step_rewards = {}
        self.active_sessions = {}
        
        # Load existing data
        self._load_data()
        
        logger.info("RAG-Gym initialized")

    def _load_data(self):
        """Load saved RAG-Gym data"""
        try:
            if self.process_trajectories_path.exists():
                with open(self.process_trajectories_path, 'r') as f:
                    self.process_trajectories = json.load(f)
                logger.info(f"Loaded {len(self.process_trajectories)} process trajectories")
        except Exception as e:
            logger.error(f"Error loading RAG-Gym data: {str(e)}")
            self.process_trajectories = []
    
    def _save_data(self):
        """Save RAG-Gym data"""
        try:
            with open(self.process_trajectories_path, 'w') as f:
                json.dump(self.process_trajectories, f)
            logger.info(f"Saved {len(self.process_trajectories)} process trajectories")
        except Exception as e:
            logger.error(f"Error saving RAG-Gym data: {str(e)}")
    
    def start_session(self, user_id, query):
        """
        Start a new RAG-Gym search session
        
        Args:
            user_id (str): User identifier
            query (str): Original query
            
        Returns:
            str: Session ID
        """
        session_id = f"{user_id}_{int(time.time())}"
        
        # Initialize session state
        self.active_sessions[session_id] = {
            "user_id": user_id,
            "original_query": query,
            "current_query": query,
            "trajectory": [],
            "step": 0,
            "start_time": datetime.now(),
            "results": [],
            "final_results": [],
            "rewards": {}
        }
        
        logger.info(f"Started RAG-Gym session {session_id} for user {user_id} with query '{query}'")
        return session_id
    
    def step_wise_search(self, session_id):
        """
        Perform RAG-Gym step-wise search with MDP-based control
        
        Args:
            session_id (str): Active session ID
            
        Returns:
            dict: Search results and process information
        """
        if session_id not in self.active_sessions:
            logger.error(f"Invalid session ID: {session_id}")
            return {"error": "Invalid session ID"}
        
        session = self.active_sessions[session_id]
        step = session["step"]
        
        # Record step start
        step_start = time.time()
        
        # Step 0: Query Understanding & Enhancement
        if step == 0:
            enhanced_query = self._enhance_query(session["original_query"], session["user_id"])
            
            session["current_query"] = enhanced_query
            session["trajectory"].append({
                "step": 0,
                "action": "query_enhancement",
                "original_query": session["original_query"],
                "enhanced_query": enhanced_query,
                "timestamp": datetime.now().isoformat()
            })
            
            session["step"] += 1
            step_time = time.time() - step_start
            self._record_step_reward(session_id, 0, "query_enhancement", step_time)
            
            return {
                "status": "in_progress",
                "step": 0,
                "action": "query_enhancement",
                "original_query": session["original_query"],
                "current_query": enhanced_query,
                "next_step": "multi_hop_retrieval"
            }
        
        # Step 1: Multi-hop Retrieval
        elif step == 1:
            multi_hop_results = self._perform_multi_hop_retrieval(session["current_query"])
            
            session["results"] = multi_hop_results
            session["trajectory"].append({
                "step": 1,
                "action": "multi_hop_retrieval",
                "query": session["current_query"],
                "result_count": len(multi_hop_results),
                "timestamp": datetime.now().isoformat()
            })
            
            session["step"] += 1
            step_time = time.time() - step_start
            self._record_step_reward(session_id, 1, "multi_hop_retrieval", step_time)
            
            return {
                "status": "in_progress",
                "step": 1,
                "action": "multi_hop_retrieval",
                "query": session["current_query"],
                "result_count": len(multi_hop_results),
                "next_step": "result_refinement"
            }
        
        # Step 2: Result Refinement & Personalization
        elif step == 2:
            refined_results = self._refine_and_personalize_results(
                session["results"], session["user_id"], session["current_query"]
            )
            
            session["final_results"] = refined_results
            session["trajectory"].append({
                "step": 2,
                "action": "result_refinement",
                "original_count": len(session["results"]),
                "refined_count": len(refined_results),
                "timestamp": datetime.now().isoformat()
            })
            
            session["step"] += 1
            step_time = time.time() - step_start
            self._record_step_reward(session_id, 2, "result_refinement", step_time)
            
            # Compute overall reward and close session
            self._compute_session_reward(session_id)
            
            # Save trajectory for learning
            self.process_trajectories.append(session["trajectory"])
            self._save_data()
            
            # Remove from active sessions
            final_results = session["final_results"]
            del self.active_sessions[session_id]
            
            return {
                "status": "complete",
                "step": 2,
                "action": "result_refinement",
                "results": final_results,
                "improved_query": session["current_query"] if session["current_query"] != session["original_query"] else None
            }
        
        else:
            logger.error(f"Invalid step {step} for session {session_id}")
            return {"error": f"Invalid step: {step}"}
    
    def _enhance_query(self, query, user_id):
        """
        Enhance query using RAG-Gym's SFT approach
        
        Args:
            query (str): Original query
            user_id (str): User identifier
            
        Returns:
            str: Enhanced query
        """
        # Step 1: Domain-specific enhancement 
        enhanced_query = self.query_agent._enhance_domain_query(query)
        
        # Step 2: Apply learned query transformations
        improved_query = self.query_agent.improve_query(enhanced_query)
        
        # Step 3: User personalization (if available)
        if self.learning_agent and user_id and user_id in self.learning_agent.user_profiles:
            user_profile = self.learning_agent.user_profiles[user_id]
            top_user_topics = sorted(user_profile.items(), key=lambda x: x[1], reverse=True)[:3]
            
            # Add user's top topics if relevant to query
            query_terms = set(query.lower().split())
            for topic, _ in top_user_topics:
                # Check if topic is at least somewhat related to query
                if any(self._terms_related(topic, term) for term in query_terms):
                    if topic not in query.lower():
                        improved_query = f"{improved_query} {topic}"
        
        logger.info(f"RAG-Gym enhanced query: '{query}' -> '{improved_query}'")
        return improved_query
    
    def _terms_related(self, term1, term2):
        """Check if two terms are related (simple implementation)"""
        if term1 == term2:
            return True
            
        # Check for substring containment
        if term1 in term2 or term2 in term1:
            return True
            
        # Could extend with word embeddings for better semantic similarity
        return False
    
    def _perform_multi_hop_retrieval(self, query):
        """
        Perform multi-hop retrieval using RAG-Gym's iterative feedback approach
        
        Args:
            query (str): Enhanced query
            
        Returns:
            list: Retrieved papers
        """
        # Use retrieval agent's multi-hop retrieval if implemented
        if hasattr(self.retrieval_agent, "multi_hop_retrieval"):
            return self.retrieval_agent.multi_hop_retrieval(query)
        
        # Fallback to standard search
        return self.retrieval_agent.search(query)
    
    def _refine_and_personalize_results(self, papers, user_id, query):
        """
        Refine and personalize results using PRM
        
        Args:
            papers (list): Retrieved papers
            user_id (str): User identifier
            query (str): Current query
            
        Returns:
            list: Refined and personalized papers
        """
        # First convert papers to dicts for processing
        paper_dicts = [self.query_agent._paper_to_dict(paper) for paper in papers]
        
        # Filter out low-quality papers (e.g., very short abstracts)
        quality_papers = [
            paper for paper in paper_dicts 
            if paper['abstract'] and len(paper['abstract']) > 50
        ]
        
        # Ensure balanced source representation
        source_balanced = self._ensure_source_balance(quality_papers)
        
        # Apply personalization if available
        if self.learning_agent and user_id != "anonymous":
            final_papers = self._apply_personalization_with_prm(source_balanced, user_id, query)
        else:
            final_papers = source_balanced
            
        logger.info(f"RAG-Gym refined results: {len(papers)} -> {len(final_papers)}")
        return final_papers
    
    def _ensure_source_balance(self, papers, min_per_source=2, max_total=30):
        """
        Ensure balanced representation from different sources
        
        Args:
            papers (list): Paper dictionaries
            min_per_source (int): Minimum papers per source
            max_total (int): Maximum total papers
            
        Returns:
            list: Balanced papers
        """
        # Group papers by source
        source_groups = defaultdict(list)
        for paper in papers:
            source = paper.get('source', 'unknown')
            source_groups[source].append(paper)
        
        # Ensure minimum representation from each source
        balanced_results = []
        for source, source_papers in source_groups.items():
            # Take minimum papers per source (or all if fewer available)
            source_count = min(min_per_source, len(source_papers))
            balanced_results.extend(source_papers[:source_count])
        
        # If we have space for more papers, add them while maintaining balance
        remaining_slots = max_total - len(balanced_results)
        if remaining_slots > 0:
            # Create lists of remaining papers by source
            remaining_by_source = {
                source: papers[min_per_source:]
                for source, papers in source_groups.items()
                if len(papers) > min_per_source
            }
            
            # Add papers in round-robin fashion until we reach max_total
            while remaining_slots > 0 and remaining_by_source:
                for source in list(remaining_by_source.keys()):
                    if not remaining_by_source[source]:
                        del remaining_by_source[source]
                        continue
                        
                    # Add one paper from this source
                    balanced_results.append(remaining_by_source[source].pop(0))
                    remaining_slots -= 1
                    
                    if remaining_slots <= 0:
                        break
        
        return balanced_results
    
    def _apply_personalization_with_prm(self, papers, user_id, query):
        """
        Apply personalization using Process Reward Modeling
        
        Args:
            papers (list): Paper dictionaries to rank
            user_id (str): User identifier
            query (str): Current query
            
        Returns:
            list: Personalized paper ranking
        """
        # Get user profile if available
        user_profile = self.learning_agent.user_profiles.get(user_id, {})
        
        # Score each paper
        scored_papers = []
        for paper in papers:
            # Base relevance score (1.0)
            base_score = 1.0
            
            # User interest score based on topic matching
            interest_score = self._calculate_user_interest_score(paper, user_profile)
            
            # Query relevance score
            relevance_score = self._calculate_query_relevance(paper, query)
            
            # Source preference score
            source_score = self._calculate_source_preference(user_id, paper.get('source', 'unknown'))
            
            # Process Reward Model score (combining all factors)
            final_score = (0.4 * relevance_score) + (0.4 * interest_score) + (0.2 * source_score)
            
            scored_papers.append((paper, final_score))
        
        # Sort by final score (descending)
        scored_papers.sort(key=lambda x: x[1], reverse=True)
        
        # Return ranked papers (without scores)
        return [paper for paper, _ in scored_papers]
    
    def _calculate_user_interest_score(self, paper, user_profile):
        """Calculate how well a paper matches user interests"""
        if not user_profile:
            return 0.5  # Neutral score if no profile
        
        # Extract paper topics (simplified)
        paper_topics = self._extract_paper_topics(paper)
        
        # Calculate match with user profile
        matching_score = 0.0
        total_weight = 0.0
        
        for topic, weight in user_profile.items():
            if topic in paper_topics:
                matching_score += weight
                total_weight += weight
        
        # Normalize
        if total_weight > 0:
            return matching_score / total_weight
        
        return 0.5  # Neutral score if no matches
    
    def _extract_paper_topics(self, paper):
        """Extract topics from paper title and abstract (simplified)"""
        topics = set()
        if paper.get('title'):
            # Simple tokenization (would use NLP techniques in production)
            topics.update(word.lower() for word in paper['title'].split() if len(word) > 3)
        
        if paper.get('abstract'):
            # Extract only significant words from abstract
            abstract_words = [word.lower() for word in paper['abstract'].split() 
                             if len(word) > 4]  # Longer words likely more meaningful
            abstract_counts = Counter(abstract_words)
            # Focus on most frequent terms
            common_terms = [word for word, count in abstract_counts.most_common(10) 
                           if count > 1]
            topics.update(common_terms)
        
        return topics
    
    def _calculate_query_relevance(self, paper, query):
        """Calculate relevance of paper to query"""
        if not query:
            return 0.5
        
        # Extract query terms
        query_terms = set(query.lower().split())
        
        # Check title for query terms
        title_terms = set(paper.get('title', '').lower().split())
        title_match = len(query_terms.intersection(title_terms)) / max(len(query_terms), 1)
        
        # Check abstract for query terms
        abstract_terms = set(paper.get('abstract', '').lower().split())
        abstract_match = len(query_terms.intersection(abstract_terms)) / max(len(query_terms), 1)
        
        # Title matches are more important than abstract matches
        return (0.7 * title_match) + (0.3 * abstract_match)
    
    def _calculate_source_preference(self, user_id, source):
        """Calculate user's preference for a source"""
        # Use learning agent's source preference calculation if available
        if hasattr(self.learning_agent, "_calculate_source_preference"):
            return self.learning_agent._calculate_source_preference(user_id, source)
        
        return 0.5  # Neutral score if no preference data
    
    def _record_step_reward(self, session_id, step, action, time_taken):
        """Record reward for a step"""
        if session_id not in self.step_rewards:
            self.step_rewards[session_id] = {}
        
        # Define base rewards for each step
        base_rewards = {
            0: 0.5,  # Query enhancement
            1: 0.5,  # Multi-hop retrieval
            2: 0.5   # Result refinement
        }
        
        # Adjust reward based on time taken (faster is better, but not too fast)
        time_factor = 1.0
        if time_taken < 0.1:
            time_factor = 0.8  # Too fast might indicate problems
        elif time_taken > 5.0:
            time_factor = 0.9  # Too slow, but still acceptable
            
        # Calculate step reward
        step_reward = base_rewards.get(step, 0.5) * time_factor
        
        # Store it
        self.step_rewards[session_id][step] = step_reward
    
    def _compute_session_reward(self, session_id):
        """Compute overall reward for session"""
        if session_id not in self.active_sessions:
            return 0.0
        
        session = self.active_sessions[session_id]
        
        # Get step rewards
        step_rewards = self.step_rewards.get(session_id, {})
        
        # Define weights for each step in overall reward
        step_weights = {0: 0.2, 1: 0.5, 2: 0.3}
        
        # Calculate weighted sum of step rewards
        total_reward = 0.0
        for step, reward in step_rewards.items():
            weight = step_weights.get(step, 0.0)
            total_reward += weight * reward
        
        # Record overall reward in session
        session["reward"] = total_reward
        
        return total_reward
    
    def record_feedback(self, user_id, query, results, clicked_papers, feedback=None):
        """
        Record user feedback for PRM training
        
        Args:
            user_id (str): User identifier
            query (str): Search query
            results (list): All returned papers
            clicked_papers (list): Papers clicked by user
            feedback (dict, optional): Explicit feedback
            
        Returns:
            float: Calculated reward
        """
        # Calculate implicit reward based on clicks
        if not results:
            return 0.0
            
        # Track clicked vs. non-clicked for preference pairs
        click_ratio = len(clicked_papers) / len(results)
        
        # Store this feedback for training
        feedback_entry = {
            "user_id": user_id,
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "result_count": len(results),
            "clicked_count": len(clicked_papers),
            "click_ratio": click_ratio,
            "explicit_feedback": feedback
        }
        
        # Save for future PRM training
        prm_data_path = self.data_dir / "prm_feedback.json"
        try:
            existing_data = []
            if prm_data_path.exists():
                with open(prm_data_path, 'r') as f:
                    existing_data = json.load(f)
                    
            existing_data.append(feedback_entry)
            
            with open(prm_data_path, 'w') as f:
                json.dump(existing_data, f)
        except Exception as e:
            logger.error(f"Error saving PRM feedback: {str(e)}")
        
        return click_ratio
    
    def train_prm_reward_model(self, epochs=10):
        """
        Train Process Reward Model using collected feedback
        
        Args:
            epochs (int): Training epochs
            
        Returns:
            bool: Success status
        """
        # In a full implementation, this would train a reward model
        # For this project, we just log that training would happen
        prm_data_path = self.data_dir / "prm_feedback.json"
        
        if not prm_data_path.exists():
            logger.warning("No PRM feedback data available for training")
            return False
        
        try:
            with open(prm_data_path, 'r') as f:
                feedback_data = json.load(f)
                
            logger.info(f"Would train PRM model with {len(feedback_data)} feedback entries for {epochs} epochs")
            
            # Save a simple version of the "model" (placeholder)
            with open(self.prm_model_path, 'wb') as f:
                pickle.dump({
                    "training_time": datetime.now().isoformat(),
                    "training_samples": len(feedback_data),
                    "epochs": epochs
                }, f)
                
            return True
        except Exception as e:
            logger.error(f"Error in PRM training: {str(e)}")
            return False
    
    def train_sft_model(self, epochs=10):
        """
        Train Supervised Fine-Tuning model for query enhancement
        
        Args:
            epochs (int): Training epochs
            
        Returns:
            bool: Success status
        """
        # Similar placeholder for SFT training
        if not self.process_trajectories:
            logger.warning("No process trajectories available for SFT training")
            return False
        
        logger.info(f"Would train SFT model with {len(self.process_trajectories)} trajectories for {epochs} epochs")
        
        # Save a simple version of the "model" (placeholder)
        with open(self.sft_model_path, 'wb') as f:
            pickle.dump({
                "training_time": datetime.now().isoformat(),
                "training_samples": len(self.process_trajectories),
                "epochs": epochs
            }, f)
            
        return True
    
    def complete_search(self, session_id):
        """
        Complete all search steps in one call
        
        Args:
            session_id (str): Session identifier
            
        Returns:
            dict: Final search results
        """
        if session_id not in self.active_sessions:
            return {"error": "Invalid session"}
        
        # Reset session to beginning
        self.active_sessions[session_id]["step"] = 0
        
        # Execute all steps
        try:
            # Step 0: Query Understanding & Enhancement
            self.step_wise_search(session_id)
            
            # Step 1: Multi-hop Retrieval
            self.step_wise_search(session_id)
            
            # Step 2: Result Refinement & Personalization
            return self.step_wise_search(session_id)
        except Exception as e:
            return {"error": str(e)}