import re

def extract_linguistic_features(headline):
    """Returns a tuple: (final_score, dictionary_of_triggers)"""
    chars = len(headline)
    words_raw = headline.split()
    if chars == 0 or len(words_raw) == 0:
        return 0.0, {}

    words_clean = re.findall(r'\b\w+\b', headline.lower())
    triggers_found = {}

    # 1. Formatting Category (Max 0.3)
    letters = [c for c in headline if c.isalpha()]
    caps = [c for c in letters if c.isupper()]
    cap_ratio = len(caps) / len(letters) if letters else 0.0
    format_score = 0.0
    
    if cap_ratio > 0.3:
        format_score += 0.15
        triggers_found['Excessive Capitalization'] = f"{round(cap_ratio*100)}% uppercase"
        
    exclamations = headline.count('!')
    questions = headline.count('?')
    if exclamations + questions > 1:
        format_score += 0.15
        triggers_found['Punctuation Abuse'] = f"Found {exclamations + questions} symbols"
        
    # 2. Structure Category (Max 0.3)
    structure_score = 0.0
    if words_raw[0].isdigit() or re.match(r'^\d+$', words_raw[0]):
        structure_score += 0.15
        triggers_found['Listicle Pattern'] = "Starts with a number"
        
    curiosity_gap = ['what happened', 'wont believe', 'won\'t believe', 'this is why', 'this one thing']
    if any(phrase in headline.lower() for phrase in curiosity_gap):
        structure_score += 0.15
        triggers_found['Curiosity Gap'] = "Withholds key information"

    # 3. Vocabulary Category (Max 0.4)
    vocab_score = 0.0
    hype_words = {'shocking', 'insane', 'unbelievable', 'secret', 'revealed', 'never', 'knew', 'genius'}
    found_hype = [w for w in words_clean if w in hype_words]
    if found_hype:
        vocab_score += 0.2
        triggers_found['Hype Words'] = f"Used: {', '.join(found_hype)}"
        
    urgency_words = {'now', 'today', 'hurry', 'late', 'need', 'must', 'watch'}
    found_urgency = [w for w in words_clean if w in urgency_words]
    if found_urgency:
        vocab_score += 0.2
        triggers_found['Urgency/Imperative'] = f"Used: {', '.join(found_urgency)}"

    final_score = min(format_score + structure_score + vocab_score, 1.0)
    
    if not triggers_found:
        triggers_found['Neutral Language'] = "No linguistic clickbait markers detected"

    return final_score, triggers_found