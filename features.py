import re

def extract_linguistic_features(headline):
    """Returns a tuple: (final_score, dictionary_of_triggers)"""
    chars = len(headline)
    words_raw = headline.split()
    if chars == 0 or len(words_raw) == 0:
        return 0.0, {}

    headline_lower = headline.lower()
    words_clean = re.findall(r'\b\w+\b', headline_lower)
    triggers_found = {}

    # 1. Formatting Category (Max 0.3)
    letters = [c for c in headline if c.isalpha()]
    caps = [c for c in letters if c.isupper()]
    cap_ratio = len(caps) / len(letters) if letters else 0.0
    format_score = 0.0

    common_acronyms = {'NASA', 'FBI', 'CIA', 'WHO', 'UN', 'EU', 'USA', 'UK', 'COVID', 'AI', 'CEO', 'GOP', 'NATO', 'IMF', 'FDA', 'CDC', 'PM', 'MP'}
    all_caps_words = [w for w in re.findall(r'\b[A-Z]{2,}\b', headline) if w not in common_acronyms]
    
    # Flag excessive caps if over 50% uppercase, or if multiple non-acronym words are ALL CAPS
    if cap_ratio > 0.50 or len(all_caps_words) >= 2:
        format_score += 0.15
        triggers_found['Excessive Capitalization'] = f"{round(cap_ratio*100)}% uppercase"
        
    exclamations = headline.count('!')
    questions = headline.count('?')
    if exclamations + questions > 1:
        format_score += 0.15
        triggers_found['Punctuation Abuse'] = f"Found {exclamations + questions} symbols"
        
    # 2. Structure Category (Max 0.3)
    structure_score = 0.0
    first_word_cleaned = re.sub(r'[^\w\s]', '', words_raw[0])
    if first_word_cleaned.isdigit():
        structure_score += 0.15
        triggers_found['Listicle Pattern'] = "Starts with a number"
        
    curiosity_gap = [
        'what happened', 'wont believe', "won't believe", 
        'this is why', 'this is how', 'this one thing', 
        'here is why', "here's why", 'see what happens', 'wait until you see'
    ]
    if any(phrase in headline_lower for phrase in curiosity_gap):
        structure_score += 0.15
        triggers_found['Curiosity Gap'] = "Withholds key information"

    # 3. Vocabulary Category (Max 0.4)
    vocab_score = 0.0
    hype_words = {
        'shocking', 'insane', 'unbelievable', 'secret', 'revealed', 
        'genius', 'mind-blowing', 'jaw-dropping', 'breathtaking', 'miracle'
    }
    found_hype = [w for w in words_clean if w in hype_words]
    if 'never knew' in headline_lower or 'you never knew' in headline_lower:
        found_hype.append('never knew')

    if found_hype:
        vocab_score += 0.2
        triggers_found['Hype Words'] = f"Used: {', '.join(set(found_hype))}"
        
    urgency_words = {'hurry', 'urgent', 'immediately', 'panic'}
    urgency_phrases = ['act now', 'right now', 'must watch', 'dont miss', "don't miss"]
    found_urgency = [w for w in words_clean if w in urgency_words]
    for phrase in urgency_phrases:
        if phrase in headline_lower:
            found_urgency.append(phrase)

    if found_urgency:
        vocab_score += 0.2
        triggers_found['Urgency/Imperative'] = f"Used: {', '.join(set(found_urgency))}"

    final_score = min(format_score + structure_score + vocab_score, 1.0)
    
    if not triggers_found:
        triggers_found['Neutral Language'] = "No linguistic clickbait markers detected"

    return final_score, triggers_found