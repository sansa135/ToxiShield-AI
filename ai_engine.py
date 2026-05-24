from transformers import pipeline

# Toxicity classifier
classifier = pipeline(
    "text-classification",
    model="unitary/toxic-bert"
)

# AI Rewrite Logic
def rewrite_comment(text):

    toxic_words = [
        "idiot",
        "stupid",
        "hate",
        "kill",
        "ugly"
    ]

    clean_text = text

    for word in toxic_words:

        clean_text = clean_text.replace(
            word,
            "***"
        )

    return clean_text

# Emotion Detection
def detect_emotion(score):

    if score > 0.8:
        return "Extreme Anger 😡"

    elif score > 0.5:
        return "Aggressive ⚠"

    elif score > 0.3:
        return "Neutral 🙂"

    else:
        return "Positive 😊"

# AI Explanation
def explain_toxicity(score):

    if score > 0.8:
        return "This comment contains highly offensive language."

    elif score > 0.5:
        return "The comment may hurt or insult others."

    else:
        return "The comment appears safe."

# Main AI Analysis
def analyze_comment(text):

    result = classifier(text)

    label = result[0]["label"]

    score = float(result[0]["score"])

    emotion = detect_emotion(score)

    explanation = explain_toxicity(score)

    rewrite = rewrite_comment(text)

    return {
        "label": label,
        "score": round(score * 100, 2),
        "emotion": emotion,
        "rewrite": rewrite,
        "explanation": explanation
    }