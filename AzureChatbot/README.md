# Campus Help Desk Chatbot + Azure AI Language

MSAI coursework, Project 2: **Connecting a Chatbot to an AI-as-a-Service**
Author: Phaneendra Yerra, University of the Cumberlands

A traditional (non-LLM) intent chatbot for a campus IT help desk, extended with
**Azure AI Language** (free F0 tier) for sentiment analysis, language detection and
key phrase extraction.

## Where these files came from

The intent classifier in `chatbot.py` (NLTK preprocessing, TF-IDF features and a
logistic regression model) and the original seven intents in `intents.json` are
**copied from my Project 1 notebook, [`TraditionalChatbot.ipynb`](../TraditionalChatbot.ipynb)**.
They were copied, not rewritten, because the goal of this project is to *extend* an
existing traditional chatbot with a cloud AI service. Changes from Project 1:

| Change | Reason |
|---|---|
| Azure AI Language sentiment, language detection and key phrases | Assignment requirement: use a cloud AI-as-a-service |
| New `capabilities` intent (`help`, `what can you do`) | Bot must list its capabilities |
| `validate()` for empty, over-long, symbol-only and gibberish input | Bot must handle malformed input |
| New training patterns, e.g. "can you reset my password", "my wifi keeps dropping" | These phrasings failed in Project 1 testing |
| Confidence threshold lowered from 0.25 to 0.20 | Project 1 evaluation showed 0.25 rejected half of the correct answers |
| Key and endpoint read from environment variables (`config.py`) | Keep secrets out of source code and GitHub |

## What the bot can do

1. Answer help desk questions: passwords, Wi-Fi, library hours, contacting support
2. List its capabilities (type `help`)
3. Detect the user's sentiment with Azure and respond with empathy when it is negative
4. Detect non-English messages and explain that it supports English only
5. Analyze any text on request: `analyze: <text>` returns sentiment, language and key phrases
6. Reject malformed input with a helpful message
7. Keep answering help desk questions if Azure is unavailable (graceful degradation)

## Files

| File | Purpose |
|---|---|
| `AzureChatbot.ipynb` | Google Colab notebook that runs everything |
| `chatbot.py` | Intent classifier, Azure client wrapper, input validation, reply logic |
| `config.py` | `DefaultConfig`: reads `MicrosoftAPIKey` and `MicrosoftAIServiceEndpoint` |
| `intents.json` | Intents, training patterns and responses |
| `app.py` | Gradio web chat interface |
| `test_chatbot.py` | 10 offline unit tests (Azure is simulated) |
| `requirements.txt` | Python dependencies |

## Setup

### 1. Create the Azure AI Language resource
1. Sign up at <https://azure.microsoft.com/en-us/free/students> with your university email.
2. In the Azure portal, search for **Language**, open **Language service**, and click **Create**.
3. Click **Continue to create your resource**. Choose your subscription, create a resource group,
   pick a region, give the resource a name, select the **Free F0** pricing tier, and accept the
   Responsible AI notice.
4. **Review + create**, then **Create**. When deployment finishes, open
   **Resource Management > Keys and Endpoint** and copy **KEY 1** and the **Endpoint**.

### 2a. Run in Google Colab (recommended)
1. Open `AzureChatbot.ipynb` in Colab.
2. Click the key icon (Secrets) and add `MicrosoftAPIKey` and `MicrosoftAIServiceEndpoint`,
   with notebook access on.
3. Run all cells.

### 2b. Run locally (Windows)
```bat
pip install -r requirements.txt
SET MicrosoftAPIKey=<your key>
SET MicrosoftAIServiceEndpoint=https://<your-resource>.cognitiveservices.azure.com/
python -m unittest test_chatbot.py -v
python app.py
```
On macOS or Linux use `export` instead of `SET`.

## Security

Never commit your key. It is read from environment variables or Colab Secrets only.
If a key is ever exposed, regenerate it on the Keys and Endpoint page.
