import unittest
from src.agents.retrieval_agent import RetrievalAgent

class TestRetrievalAgent(unittest.TestCase):

    def setUp(self):
        self.agent = RetrievalAgent()
        self.agent.initialize()

    def test_search_research_papers(self):
        query = "machine learning"
        results = self.agent.search(query)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

    def test_empty_query(self):
        query = ""
        results = self.agent.search(query)
        self.assertEqual(results, [])

    def test_invalid_query(self):
        query = None
        with self.assertRaises(TypeError):
            self.agent.search(query)

    def test_retrieval_accuracy(self):
        query = "deep learning"
        results = self.agent.search(query)
        # Assuming we have a way to check the relevance of results
        self.assertTrue(all("deep learning" in paper.title.lower() for paper in results))

if __name__ == '__main__':
    unittest.main()