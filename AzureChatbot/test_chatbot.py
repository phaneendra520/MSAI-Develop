"""Offline tests: the Azure client is replaced by a fake, so no key is needed.

Run with:  python -m unittest test_chatbot.py -v
"""
import unittest
from types import SimpleNamespace as NS

from chatbot import AzureLanguage, Chatbot


class FakeTextAnalytics:
    """Mimics the parts of TextAnalyticsClient the bot uses."""

    def __init__(self, sentiment="neutral", neg=0.1, lang=("English", "en", 1.0), fail=False):
        self.sentiment, self.neg, self.lang, self.fail = sentiment, neg, lang, fail

    def _check(self):
        if self.fail:
            raise ConnectionError("simulated network failure")

    def detect_language(self, docs):
        self._check()
        n, c, s = self.lang
        return [NS(is_error=False, primary_language=NS(name=n, iso6391_name=c, confidence_score=s))]

    def analyze_sentiment(self, docs):
        self._check()
        pos = 0.9 if self.sentiment == "positive" else 0.05
        scores = NS(positive=pos, neutral=max(0.0, 1 - pos - self.neg), negative=self.neg)
        return [NS(is_error=False, sentiment=self.sentiment, confidence_scores=scores)]

    def extract_key_phrases(self, docs):
        self._check()
        return [NS(is_error=False, key_phrases=["library hours"])]


def make_bot(**kw):
    return Chatbot("intents.json", azure=AzureLanguage(client=FakeTextAnalytics(**kw)))


class TestChatbot(unittest.TestCase):
    def test_capabilities_list(self):
        for q in ["help", "what can you do", "what are your capabilities"]:
            self.assertIn("Here is what I can do", make_bot().reply(q), q)

    def test_multiple_prompts(self):
        bot = make_bot()
        self.assertIn("self-service portal", bot.reply("I forgot my password"))
        self.assertIn("campus network", bot.reply("wifi is not working"))
        self.assertIn("library is open", bot.reply("library hours"))
        self.assertIn("helpdesk@", bot.reply("how do I contact support"))

    def test_malformed_input(self):
        bot = make_bot()
        self.assertIn("empty", bot.reply(""))
        self.assertIn("empty", bot.reply("    "))
        self.assertIn("empty", bot.reply(None))
        self.assertIn("numbers or symbols", bot.reply("12345 !!! ???"))
        self.assertIn("doesn't look like words", bot.reply("asdfgh qwrtz"))
        self.assertIn("characters long", bot.reply("a" * 600))

    def test_out_of_scope_fallback(self):
        self.assertIn("didn't understand", make_bot().reply("what is the weather today"))

    def test_negative_sentiment_adds_empathy(self):
        bot = make_bot(sentiment="negative", neg=0.95)
        r = bot.reply("I forgot my password and I am so frustrated")
        self.assertTrue(r.startswith("I'm sorry you're dealing with this."), r)
        self.assertIn("negative sentiment", r)

    def test_neutral_has_no_empathy_prefix(self):
        r = make_bot().reply("I forgot my password")
        self.assertFalse(r.startswith("I'm sorry"))

    def test_non_english_detected(self):
        bot = make_bot(lang=("Spanish", "es", 1.0))
        self.assertIn("Spanish", bot.reply("olvidé mi contraseña por favor"))

    def test_analyze_command(self):
        r = make_bot(sentiment="positive").reply("analyze: The new library hours are fantastic")
        self.assertIn("Sentiment: positive", r)
        self.assertIn("Key phrases: library hours", r)

    def test_azure_failure_degrades_gracefully(self):
        bot = make_bot(fail=True)
        self.assertIn("self-service portal", bot.reply("I forgot my password"))
        self.assertIn("not available", bot.reply("analyze: hello world"))
        self.assertIn("ConnectionError", bot.azure.last_error)

    def test_runs_without_azure_configured(self):
        bot = Chatbot("intents.json", azure=AzureLanguage(key="", endpoint=""))
        self.assertFalse(bot.azure.available)
        self.assertIn("self-service portal", bot.reply("reset my password"))


if __name__ == "__main__":
    unittest.main()
