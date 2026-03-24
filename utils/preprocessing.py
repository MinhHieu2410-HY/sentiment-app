"""
Preprocessing utilities — mirror of DoAn_II___.ipynb
clean_text_basic → lemmatize_word → preprocess_text pipeline
"""
import re

# --------------------------------------------------------------------------- #
# Stopwords (identical to notebook)
# --------------------------------------------------------------------------- #
ENGLISH_STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before
being below between both but by can did do does doing down during each few for from
further get go had has have having he her here hers herself him himself his how i if
in into is it its itself just me more most my myself of off on once other our ours
ourselves out over own re same she should so some such than that the their theirs them
themselves then there these they this those through to too under until up us very
was we were what when where which while who whom why will with you your yours yourself
yourselves s t d ll m re ve y
""".split())


# --------------------------------------------------------------------------- #
# Text cleaning helpers
# --------------------------------------------------------------------------- #
def clean_text_basic(text: str) -> str:
    """Lowercase, remove URLs and non-alpha characters."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)   # remove URLs
    text = re.sub(r'[^a-z\s]', '', text)          # keep only letters + spaces
    text = re.sub(r'\s+', ' ', text).strip()       # collapse whitespace
    return text


def lemmatize_word(word: str) -> str:
    """Simple rule-based lemmatiser (identical to notebook)."""
    if len(word) <= 3:
        return word
    if word.endswith('ies') and len(word) > 4:
        return word[:-3] + 'y'
    if word.endswith('ied') and len(word) > 4:
        return word[:-3] + 'y'
    if word.endswith('ing') and len(word) > 6:
        return word[:-3]
    if word.endswith('ness') and len(word) > 6:
        return word[:-4]
    if word.endswith('tion') and len(word) > 6:
        return word[:-3]
    if word.endswith('ly') and len(word) > 5:
        return word[:-2]
    if word.endswith('ers') and len(word) > 5:
        return word[:-1]
    return word


def preprocess_text(text: str) -> str:
    """Full pipeline: clean → remove stopwords → lemmatise."""
    clean = clean_text_basic(text)
    tokens = [
        lemmatize_word(w)
        for w in clean.split()
        if w not in ENGLISH_STOPWORDS and len(w) > 2
    ]
    return ' '.join(tokens)


# --------------------------------------------------------------------------- #
# Label helper
# --------------------------------------------------------------------------- #
def score_to_sentiment(score) -> str:
    """Map 1-5 star score to sentiment label."""
    try:
        s = int(float(score))
    except (ValueError, TypeError):
        return 'neutral'
    if s in (1, 2):
        return 'negative'
    elif s == 3:
        return 'neutral'
    else:
        return 'positive'
