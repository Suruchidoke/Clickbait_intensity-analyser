import re

def extract_linguistic_score(headline):
    """
    Analyzes a headline for clickbait linguistic markers and 
    returns a normalized score between 0.0 and 1.0.
    """
    chars = len(headline)
    words = headline.split()
    word_count = len(words)
    
    if chars == 0 or word_count == 0:
        return 0.0

    # Feature 1: Capitalization ratio (uppercase letters vs total letters)
    letters = [c for c in headline if c.isalpha()]
    caps = [c for c in letters if c.isupper()]
    cap_ratio = len(caps) / len(letters) if letters else 0.0

    # Feature 2: Punctuation abuse (scales up to a max penalty)
    exclamation_count = headline.count('!')
    question_count = headline.count('?')
    punct_score = min((exclamation_count + question_count) * 0.25, 1.0) 

    # Feature 3: Listicles (Starts with a number)
    starts_with_num = 1.0 if words[0].isdigit() or re.match(r'^\d+$', words[0]) else 0.0

    # Feature 4: High-intensity trigger words
    triggers = {'why', 'how', 'this', 'these', 'what', 'unbelievable', 'shocking', 'secret', 'revealed', 'never', 'knew'}
    trigger_count = sum(1 for word in words if word.lower() in triggers)
    trigger_score = min(trigger_count * 0.3, 1.0)

    # Aggregate weighted score (sums to max 1.0)
    final_score = (
        (cap_ratio * 0.25) + 
        (punct_score * 0.25) + 
        (starts_with_num * 0.25) + 
        (trigger_score * 0.25)
    )
    
    return min(final_score, 1.0)

# Local testing block
if __name__ == "__main__":
    test_headlines = [
        "Mumbai University Announces 2026 Examination Schedule",
        "7 Things You Never Knew About Your Phone!",
        "WHY THIS IS AMAZING!!!"
    ]
    
    print("--- LINGUISTIC SCORING ENGINE ---")
    for h in test_headlines:
        score = extract_linguistic_score(h)
        print(f"Score: {score:.2f} | {h}")