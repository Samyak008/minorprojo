from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import time
import logging
from datetime import datetime

class BaseAgent(ABC):
    """
    Enhanced Base Agent with support for memory, metrics, and RAG integration
    
    This abstract base class defines the interface for all agents in the system.
    It includes support for agent memory, performance metrics, and integration
    with retrieval-augmented generation (RAG) systems.
    """
    
    def __init__(self, name: str = None):
        """
        Initialize the base agent
        
        Args:
            name (str, optional): Name of this agent instance
        """
        self.name = name or self.__class__.__name__
        self.memory = {}  # Agent's memory/state
        self.metrics = {
            "queries_processed": 0,
            "successful_responses": 0,
            "failed_responses": 0,
            "avg_response_time": 0,
            "total_response_time": 0
        }
        self.creation_time = datetime.now()
        self.logger = logging.getLogger(f"agent.{self.name}")
    
    @abstractmethod
    def initialize(self):
        """Initialize the agent with necessary resources and data"""
        pass

    @abstractmethod
    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute the agent's primary function
        
        Args:
            query (str): The query or instruction to process
            context (dict, optional): Additional context for processing
            
        Returns:
            Any: The result of processing the query
        """
        start_time = time.time()
        self.metrics["queries_processed"] += 1
        
        try:
            # Child classes implement this logic
            result = None  # Placeholder
            
            # Update metrics
            self.metrics["successful_responses"] += 1
            execution_time = time.time() - start_time
            self.metrics["total_response_time"] += execution_time
            self.metrics["avg_response_time"] = (
                self.metrics["total_response_time"] / 
                self.metrics["successful_responses"]
            )
            
            return result
        except Exception as e:
            self.metrics["failed_responses"] += 1
            self.logger.error(f"Error executing query: {str(e)}")
            raise

    @abstractmethod
    def respond(self) -> Dict[str, Any]:
        """
        Format and return the agent's response
        
        Returns:
            dict: The formatted response
        """
        pass
    
    def remember(self, key: str, value: Any) -> None:
        """
        Store information in the agent's memory
        
        Args:
            key (str): Memory key
            value (Any): Value to remember
        """
        self.memory[key] = value
    
    def recall(self, key: str, default: Any = None) -> Any:
        """
        Retrieve information from the agent's memory
        
        Args:
            key (str): Memory key
            default (Any, optional): Default value if key not found
            
        Returns:
            Any: The remembered value or default
        """
        return self.memory.get(key, default)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get the agent's performance metrics
        
        Returns:
            dict: Performance metrics
        """
        return self.metrics
    
    def reset(self) -> None:
        """Reset the agent's state while preserving learned knowledge"""
        self.memory = {}
        
    def get_status(self) -> Dict[str, Any]:
        """
        Get the agent's current status
        
        Returns:
            dict: Agent status information
        """
        return {
            "name": self.name,
            "uptime": (datetime.now() - self.creation_time).total_seconds(),
            "metrics": self.metrics,
            "memory_keys": list(self.memory.keys())
        }
    
    # RAG Integration Methods
    def retrieve_knowledge(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant knowledge for the query (for RAG integration)
        
        Args:
            query (str): The query to find knowledge for
            limit (int): Maximum number of knowledge pieces to retrieve
            
        Returns:
            list: Relevant knowledge pieces
        """
        # Default implementation returns empty list
        # Override in concrete implementations
        return []
    
    def generate_response(self, query: str, knowledge: List[Dict[str, Any]]) -> str:
        """
        Generate a response using retrieved knowledge (for RAG integration)
        
        Args:
            query (str): The user query
            knowledge (list): Retrieved knowledge pieces
            
        Returns:
            str: Generated response
        """
        # Default implementation returns empty string
        # Override in concrete implementations
        return ""