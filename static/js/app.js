// Global state
let allPapers = [];
let currentQuery = "";
let userId = localStorage.getItem('research_user_id') || 'user_' + Math.random().toString(36).substring(2, 15);
localStorage.setItem('research_user_id', userId);

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('search-button').addEventListener('click', searchPapers);
    document.getElementById('search-input').addEventListener('keypress', e => {
        if (e.key === 'Enter') searchPapers();
    });
});

async function searchPapers() {
    const query = document.getElementById('search-input').value.trim();
    if (!query) {
        alert('Please enter a search query.');
        return;
    }

    document.getElementById('loading').style.display = 'block';
    document.getElementById('results-container').style.display = 'none';

    try {
        const response = await fetch(`/search/?query=${encodeURIComponent(query)}&user_id=${userId}`, {
            method: 'POST'
        });

        if (!response.ok) {
            throw new Error('Failed to fetch search results.');
        }

        const data = await response.json();
        allPapers = data.results || [];
        displayResults(allPapers);
        document.getElementById('results-count').textContent = `Found ${allPapers.length} papers`;
    } catch (error) {
        console.error('Error fetching search results:', error);
        alert('Error searching papers. Please try again.');
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
}

function displayResults(papers) {
    const resultsContainer = document.getElementById('results');
    resultsContainer.innerHTML = ''; // Clear previous results

    if (papers.length === 0) {
        resultsContainer.innerHTML = '<p>No results found.</p>';
        return;
    }

    papers.forEach(paper => {
        const paperElement = document.createElement('div');
        paperElement.className = 'paper';

        paperElement.innerHTML = `
            <h3><a href="${paper.url}" target="_blank" class="paper-link">${paper.title}</a></h3>
            <p class="authors">${paper.authors.join(', ')}</p>
            <p class="abstract">${paper.abstract}</p>
            <span class="source-badge">${paper.source}</span>
        `;

        resultsContainer.appendChild(paperElement);
    });

    document.getElementById('results-container').style.display = 'block';
}

function filterResults() {
    const sourceFilter = document.getElementById('filter-source').value;

    if (sourceFilter === 'all') {
        displayResults(allPapers);
        document.getElementById('results-count').textContent = `Found ${allPapers.length} papers`;
    } else {
        const filtered = allPapers.filter(paper => paper.source === sourceFilter);
        displayResults(filtered);
        document.getElementById('results-count').textContent = `Found ${filtered.length} papers from ${sourceFilter}`;
    }
}