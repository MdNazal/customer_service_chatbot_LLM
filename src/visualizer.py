import re
from collections import Counter

# Common words to exclude from visualization
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "we", "our", "us", "i", "it", "its", "as", "we",
    "which", "who", "whom", "when", "where", "why", "how", "all", "each",
    "both", "few", "more", "most", "other", "some", "such", "than", "too",
    "very", "just", "also", "into", "through", "during", "paper", "show",
    "present", "propose", "method", "approach", "using", "based", "two",
    "new", "used", "use", "while", "their", "they", "them", "then", "than",
    "well", "not", "proposed", "results", "show", "shows", "shown", "between"
}


def extract_keywords(texts, top_n=20):
    """Extract top keywords from a list of texts."""
    combined = " ".join(texts).lower()
    words = re.findall(r'\b[a-z]{4,}\b', combined)
    filtered = [w for w in words if w not in STOPWORDS]
    counter = Counter(filtered)
    return counter.most_common(top_n)


def get_category_labels(categories_list):
    """Convert arXiv category codes to readable labels."""
    category_map = {
        "cs.AI": "Artificial Intelligence",
        "cs.LG": "Machine Learning",
        "cs.CV": "Computer Vision",
        "cs.CL": "Computation & Language (NLP)",
        "cs.NE": "Neural & Evolutionary Computing",
        "cs.RO": "Robotics",
        "cs.CR": "Cryptography & Security",
        "cs.DB": "Databases",
        "cs.DC": "Distributed Computing",
        "cs.DS": "Data Structures & Algorithms",
        "cs.HC": "Human-Computer Interaction",
        "cs.IR": "Information Retrieval",
        "cs.IT": "Information Theory",
        "cs.MA": "Multiagent Systems",
        "cs.NI": "Networking",
        "cs.OS": "Operating Systems",
        "cs.PL": "Programming Languages",
        "cs.SE": "Software Engineering",
        "cs.SY": "Systems & Control",
    }

    labels = []
    for cats in categories_list:
        for cat in cats.split():
            if cat in category_map:
                labels.append(category_map[cat])
    return labels


def build_bar_chart_data(keywords):
    """Convert keyword list to chart-ready format."""
    words = [k[0] for k in keywords]
    counts = [k[1] for k in keywords]
    return words, counts
