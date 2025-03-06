from setuptools import setup, find_packages

setup(
    name='research-paper-retrieval-system',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A multi-agent AI system for retrieving research papers based on user queries.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'flask',  # or 'fastapi'
        'torch',  # or 'tensorflow'
        'numpy',
        'scikit-learn',
        'faiss-cpu',  # or 'faiss-gpu' if using GPU
        'sentence-transformers',  # for text embeddings
        'whoosh',  # for BM25 implementation
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)