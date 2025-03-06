def parse_query(query):
    # Normalize the query by converting to lowercase and stripping whitespace
    normalized_query = query.lower().strip()
    return normalized_query

def extract_keywords(query):
    # Split the query into keywords based on whitespace
    keywords = normalized_query.split()
    return keywords

def format_query_for_search(keywords):
    # Join keywords with a space for search compatibility
    formatted_query = ' '.join(keywords)
    return formatted_query

def parse_and_format_query(query):
    normalized_query = parse_query(query)
    keywords = extract_keywords(normalized_query)
    formatted_query = format_query_for_search(keywords)
    return formatted_query