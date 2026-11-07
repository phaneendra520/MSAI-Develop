"""Traditional intent chatbot extended with Azure AI Language.

ORIGIN: The intent classifier (NLTK preprocessing, TF-IDF features and a
logistic regression model) is copied from my Project 1 notebook,
TraditionalChatbot.ipynb. It is copied rather than rewritten because this
project's goal is to extend an existing traditional chatbot with a cloud
AI-as-a-service capability, not to rebuild the classifier.

NEW IN THIS PROJECT:
  * Azure AI Language (Text Analytics) sentiment analysis, language
    detection and key phrase extraction on every message
  * a "capabilities" intent that lists what the bot can do
  * validation of malformed input (empty, too long, symbols only, gibberish)
  * graceful degradation when Azure is unavailable or not configured
"""
import json
import random
import re

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from config import DefaultConfig

for _pkg in ["punkt", "punkt_tab", "wordnet", "omw-1.4"]:
    nltk.download(_pkg, quiet=True)

CAPABILITIES = (
    "Here is what I can do:\n"
    "1. Help with a forgotten or locked password\n"
    "2. Troubleshoot Wi-Fi and internet problems\n"
    "3. Tell you the library hours\n"
    "4. Give you help desk contact details\n"
    "5. Detect how you are feeling (using Azure AI Language sentiment analysis) "
    "and adjust my reply\n"
    "6. Analyze any text for you: type  analyze: <your text>  to get its "
    "sentiment, language and key phrases\n"
    "Type  help  at any time to see this list again."
)

_lem = WordNetLemmatizer()


def clean(text: str) -> str:
    """Lowercase, strip non-letters, tokenize and lemmatize (from Project 1)."""
    text = re.sub(r"[^a-z\s]", "", text.lower())
    return " ".join(_lem.lemmatize(w) for w in nltk.word_tokenize(text))


# --------------------------------------------------------------------------
# Azure AI Language client
# --------------------------------------------------------------------------
class AzureLanguage:
    """Thin wrapper around TextAnalyticsClient that never raises to the bot."""

    def __init__(self, key: str = None, endpoint: str = None, client=None):
        self.client = client
        self.last_error = None
        key = key if key is not None else DefaultConfig.API_KEY
        endpoint = endpoint if endpoint is not None else DefaultConfig.ENDPOINT_URI
        if self.client is None and key and endpoint:
            from azure.ai.textanalytics import TextAnalyticsClient
            from azure.core.credentials import AzureKeyCredential
            self.client = TextAnalyticsClient(endpoint=endpoint,
                                              credential=AzureKeyCredential(key))

    @property
    def available(self) -> bool:
        return self.client is not None

    def analyze(self, text: str, key_phrases: bool = False):
        """Return sentiment, confidence scores, language and (optionally) key phrases.

        Returns None if Azure is not configured or the call fails, so the
        chatbot can keep working without the AI service.
        """
        if not self.available:
            return None
        try:
            lang = self.client.detect_language([text])[0]
            sent = self.client.analyze_sentiment([text])[0]
            if lang.is_error or sent.is_error:
                self.last_error = getattr(sent, "error", None) or getattr(lang, "error", None)
                return None
            result = {
                "sentiment": sent.sentiment,
                "scores": {
                    "positive": sent.confidence_scores.positive,
                    "neutral": sent.confidence_scores.neutral,
                    "negative": sent.confidence_scores.negative,
                },
                "language": lang.primary_language.name,
                "language_code": lang.primary_language.iso6391_name,
                "language_confidence": lang.primary_language.confidence_score,
            }
            if key_phrases:
                kp = self.client.extract_key_phrases([text])[0]
                result["key_phrases"] = [] if kp.is_error else list(kp.key_phrases)
            self.last_error = None
            return result
        except Exception as exc:  # network, authentication, quota errors
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None


# --------------------------------------------------------------------------
# Chatbot
# --------------------------------------------------------------------------
class Chatbot:
    SUPPORT_INTENTS = {"password_reset", "wifi_help", "contact_support", "library_hours"}

    def __init__(self, intents_path: str = "intents.json", azure: AzureLanguage = None,
                 threshold: float = DefaultConfig.CONFIDENCE_THRESHOLD,
                 show_analysis: bool = True):
        with open(intents_path) as f:
            data = json.load(f)
        X, y = [], []
        for it in data["intents"]:
            for p in it["patterns"]:
                X.append(clean(p))
                y.append(it["tag"])
        self.vec = TfidfVectorizer(ngram_range=(1, 2))
        self.clf = LogisticRegression(max_iter=1000).fit(self.vec.fit_transform(X), y)
        self.responses = {it["tag"]: it["responses"] for it in data["intents"]}
        self.vocab = {w for s in X for w in s.split()}
        self.azure = azure if azure is not None else AzureLanguage()
        self.threshold = threshold
        self.show_analysis = show_analysis

    # ---- intent classification (Project 1) ----
    def classify(self, msg: str):
        probs = self.clf.predict_proba(self.vec.transform([clean(msg)]))[0]
        i = probs.argmax()
        return self.clf.classes_[i], float(probs[i])

    # ---- malformed input handling ----
    def validate(self, msg):
        """Return an error reply for malformed input, or None if the input is usable."""
        if msg is None or not str(msg).strip():
            return "It looks like your message was empty. Type a question, or type  help  to see what I can do."
        msg = str(msg)
        if len(msg) > DefaultConfig.MAX_INPUT_CHARS:
            return (f"That message is {len(msg)} characters long. Please keep questions under "
                    f"{DefaultConfig.MAX_INPUT_CHARS} characters so I can understand them.")
        if not re.search(r"[A-Za-z]", msg):
            return "I can only read questions written in words, not just numbers or symbols. Type  help  to see what I can do."
        words = re.findall(r"[a-z]+", msg.lower())
        long_words = [w for w in words if len(w) > 2]
        if long_words and not any(
                w in self.vocab or _lem.lemmatize(w) in self.vocab or wordnet.synsets(w)
                for w in long_words):
            return "Sorry, that doesn't look like words I recognize. Could you rephrase? Type  help  to see what I can do."
        return None

    # ---- reply composition ----
    def _analysis_note(self, a):
        if not a or not self.show_analysis:
            return ""
        top = a["scores"][a["sentiment"]] if a["sentiment"] in a["scores"] else max(a["scores"].values())
        return f"\n\n[Azure AI Language: {a['sentiment']} sentiment, {top:.0%} confidence]"

    def _analyze_command(self, text):
        if not text.strip():
            return "Please put some text after  analyze:  for example  analyze: I love the new library."
        problem = self.validate(text)
        if problem:
            return problem
        a = self.azure.analyze(text, key_phrases=True)
        if a is None:
            return ("The Azure AI Language service is not available right now, so I can't analyze "
                    "that text. I can still answer help desk questions.")
        s = a["scores"]
        phrases = ", ".join(a.get("key_phrases") or []) or "none found"
        return (f"Text analysis from Azure AI Language:\n"
                f"Sentiment: {a['sentiment']} (positive {s['positive']:.2f}, neutral {s['neutral']:.2f}, "
                f"negative {s['negative']:.2f})\n"
                f"Language: {a['language']} ({a['language_confidence']:.0%} confidence)\n"
                f"Key phrases: {phrases}")

    def reply(self, msg) -> str:
        problem = self.validate(msg)
        if problem:
            return problem
        msg = str(msg).strip()

        m = re.match(r"^\s*analy[sz]e\s*:?\s*(.*)$", msg, re.IGNORECASE | re.DOTALL)
        if m:
            return self._analyze_command(m.group(1))

        a = self.azure.analyze(msg)

        # Azure language detection: politely decline non-English messages.
        if (a and a["language_code"] not in ("en", "(Unknown)")
                and a["language_confidence"] >= 0.9 and len(msg.split()) >= 3):
            return (f"It looks like you wrote in {a['language']}. I currently understand English only. "
                    f"Type  help  to see what I can do.")

        intent, conf = self.classify(msg)
        negative = bool(a and a["sentiment"] == "negative"
                        and a["scores"]["negative"] >= DefaultConfig.NEGATIVE_SENTIMENT_CUTOFF)

        if conf < self.threshold:
            if negative:
                text = ("I can tell this is frustrating, and I'm sorry I didn't understand. "
                        "Could you rephrase? Type  help  to see what I can do, or ask for a human agent.")
            else:
                text = "Sorry, I didn't understand. Could you rephrase? Type  help  to see what I can do."
            return text + self._analysis_note(a)

        if intent == "capabilities":
            text = CAPABILITIES
        else:
            text = random.choice(self.responses[intent])
            if negative and intent in self.SUPPORT_INTENTS:
                text = "I'm sorry you're dealing with this. " + text
            elif negative and intent == "goodbye":
                text = ("I'm sorry I couldn't do more for you today. If you still need help, "
                        "the help desk is at helpdesk@example.edu. ") + text
        return text + self._analysis_note(a)


if __name__ == "__main__":
    bot = Chatbot()
    print("Azure configured:", bot.azure.available)
    print("Bot: Hi! Type  help  to see what I can do, or  quit  to exit.")
    while True:
        user = input("You: ")
        if user.strip().lower() in {"quit", "exit"}:
            break
        print("Bot:", bot.reply(user))
