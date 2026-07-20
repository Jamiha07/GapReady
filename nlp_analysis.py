"""
NLP layer for GapReady.
Turns the resume and JD into TF-IDF vectors, then produces:
  1. coverage_score  - % of the JD's requirement terms found in the resume (headline score)
  2. similarity      - plain cosine similarity between the two documents (secondary stat)
  3. candidate_gaps  - JD terms NOT found in the resume (sent to the LLM for filtering)
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# JD-filler words that are not skills. Without this, words like "intern" or
# "requirements" would show up as fake requirements in every analysis.
EXTRA_STOPWORDS = {
    "intern", "internship", "requirements", "required", "requirement",
    "experience", "role", "responsibilities", "team", "work", "working",
    "strong", "familiarity", "understanding", "knowledge", "ability",
    "skills", "plus", "nice", "location", "duration", "months", "similar",
    "basic", "prior", "using", "existing", "looking", "motivated", "join",
}
ALL_STOPWORDS = set(ENGLISH_STOP_WORDS) | EXTRA_STOPWORDS


def clean_text(text):
    """Normalize text so word matching is fair:
    lowercase everything, remove punctuation (but keep + # . so 'c++', 'c#',
    '.net' survive), and collapse repeated whitespace into single spaces."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\+\#\.]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def analyze(resume_text, jd_text):
    """Main entry point. Returns a dictionary with the outputs listed at the top."""
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)

    # --- Part A: overall document similarity (the secondary stat) ---
    # Here two-word phrases (ngram_range 1,2) are fine, because we're comparing
    # whole documents, not building a requirements list.
    doc_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = doc_vectorizer.fit_transform([resume_clean, jd_clean])
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

    # --- Part B: build the JD requirements list ---
    # Single words only. Why not phrases? Testing showed phrases create junk
    # ("containerization aws") and redundancy (aws / cloud / aws cloud) that
    # distorts the score. Single words = clean skill terms.
    # Why not "top N by TF-IDF weight"? With only 2 documents, IDF punishes
    # exactly the terms that appear in BOTH docs - i.e. the matches - so the
    # "most important" terms would all be misses. Frequency-based selection
    # from the JD alone avoids that trap.
    jd_words = []
    for word in jd_clean.split():
        word = word.strip(".")   # drop sentence-ending dots ("intern." -> "intern") but keep ".net" intact inside words
        if word not in ALL_STOPWORDS and len(word) > 1 and not word.isdigit():
            if word not in jd_words:          # keep first occurrence order, no duplicates
                jd_words.append(word)

    # --- Part C: coverage check (one-directional) ---
    # A JD word counts as covered if it appears as a whole word in the resume.
    # Whole-word matching (via a set) prevents cheats like "java" matching
    # inside "javascript". Extra resume skills are simply never looked at,
    # so they can't lower the score.
    resume_words = set(w.strip(".") for w in resume_clean.split())
    covered = [w for w in jd_words if w in resume_words]
    missing = [w for w in jd_words if w not in resume_words]

    coverage_score = (len(covered) / len(jd_words)) * 100 if jd_words else 0

    return {
        "coverage_score": round(coverage_score, 1),
        "similarity": round(similarity * 100, 1),
        "candidate_gaps": missing,
        "covered_terms": covered,
    }


# --- Standalone test: run "python nlp_analysis.py" to see it work
#     on sample text BEFORE wiring it into the Flask app ---
if __name__ == "__main__":
    sample_resume = """
    Jamiha Tashfeen. BSCS student at NUST. Skills: Python, Flask, JavaScript,
    HTML, CSS, MySQL, Git, GitHub, REST API development, prompt engineering,
    LLM integration with Groq API. Projects: GapReady interview prep tool,
    Semester Sage academic planner using Gemini API.
    """
    sample_jd = """
    Full Stack Intern. Requirements: JavaScript, Python, REST API design,
    MySQL or PostgreSQL databases, Git version control, Docker containerization,
    AWS cloud deployment, OAuth authentication.
    """

    result = analyze(sample_resume, sample_jd)
    print("Coverage score:", result["coverage_score"], "%")
    print("Overall similarity:", result["similarity"], "%")
    print("Covered terms:", result["covered_terms"])
    print("Candidate gaps:", result["candidate_gaps"])


# ---------------------------------------------------------------------------
# K-Means skill clustering
# Groups the final gap list into related categories for organized display.
# Runs entirely locally (no API), deterministic thanks to random_state.
# ---------------------------------------------------------------------------
from sklearn.cluster import KMeans


def cluster_gaps(gap_phrases):
    """
    Takes the final list of gap strings (e.g. "Copywriting skills",
    "Canva or Adobe Creative Suite") and groups related ones together.

    Returns a list of groups: [{"label": "...", "items": [...]}, ...]
    or None when clustering wouldn't help (fewer than 4 gaps) —
    the frontend then just shows the flat list as before.
    """
    # Too few gaps -> grouping is pointless noise. Keep the flat list.
    if not gap_phrases or len(gap_phrases) < 4:
        return None

    # 1. Turn each gap phrase into a TF-IDF vector.
    #    Fitting on the phrases themselves means words shared between phrases
    #    (e.g. "skills", "tools") pull those phrases toward the same cluster.
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        vectors = vectorizer.fit_transform(gap_phrases)
    except ValueError:
        # every phrase was stopwords-only — can't vectorize, keep flat list
        return None

    # 2. Choose how many groups: roughly one group per 2-3 gaps, capped at 4.
    #    (4 gaps -> 2 groups, 8 gaps -> 3, 10+ -> 4)
    n_clusters = min(4, max(2, len(gap_phrases) // 3))

    # 3. Run K-Means. random_state pins the randomness so the same input
    #    always produces the same grouping (deterministic = defensible).
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(vectors)

    # 4. Name each group after its most characteristic word: the term with
    #    the highest weight in the cluster's centre. K-Means itself can't
    #    name clusters — this is the standard trick for readable labels.
    feature_names = vectorizer.get_feature_names_out()
    groups = []
    for cluster_id in range(n_clusters):
        items = [gap_phrases[i] for i in range(len(gap_phrases)) if labels[i] == cluster_id]
        if not items:
            continue
        centre = km.cluster_centers_[cluster_id]
        top_word = feature_names[centre.argmax()]
        label = top_word.capitalize() + " related"
        groups.append({"label": label, "items": items})

    # Tidy-up: merge single-item groups into one "Other" group so the display
    # never fragments into many one-line headings.
    multi = [g for g in groups if len(g["items"]) > 1]
    singles = [g for g in groups if len(g["items"]) == 1]
    if len(singles) >= 2:
        multi.append({"label": "Other requirements",
                      "items": [g["items"][0] for g in singles]})
        groups = multi
    elif singles and multi:
        groups = multi + singles

    # If K-Means degenerated into one big group, the flat list reads better.
    if len(groups) < 2:
        return None
    return groups


if __name__ == "__main__" and True:
    # quick standalone test for the clustering:  python nlp_analysis.py
    sample_gaps = [
        "Marketing knowledge",
        "Copywriting skills",
        "Social media campaign execution",
        "Influencer outreach and partnership coordination",
        "Graphic design and video editing",
        "Familiarity with Canva or Adobe Creative Suite",
        "Campaign metrics tracking and reporting",
        "Experience with Meta Ads Manager or Google Analytics",
    ]
    print("\n--- K-Means clustering test ---")
    for g in cluster_gaps(sample_gaps) or []:
        print(f"[{g['label']}]")
        for item in g["items"]:
            print("  -", item)