"""
NLP Chatbot — Unlimited Knowledge Edition
==========================================
Removes all topic limitations by adding:
  1. Expanded built-in knowledge (100+ facts across 15+ topics)
  2. Wikipedia auto-fetch  — bot pulls live Wikipedia summaries on demand
  3. LLM fallback          — uses OpenAI/Anthropic API for anything else
  4. All previous features retained (history, teach, intents, semantic search)

Install:
    pip install nltk scikit-learn numpy sentence-transformers wikipedia-api openai
"""

import nltk, numpy as np, random, string, warnings, json, os, re
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

for pkg in ["punkt", "wordnet", "stopwords", "averaged_perceptron_tagger", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Optional: Sentence-Transformers ──────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    SEMANTIC_AVAILABLE = True
    print("[✔] Sentence-Transformers loaded.")
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("[!] sentence-transformers not found — using TF-IDF.")

# ── Optional: Wikipedia ───────────────────────────────────────────────────────
try:
    import wikipediaapi
    _WIKI = wikipediaapi.Wikipedia(
        language="en",
        user_agent="NLPBot/2.0 (educational chatbot project)"
    )
    WIKI_AVAILABLE = True
    print("[✔] Wikipedia API loaded.")
except ImportError:
    WIKI_AVAILABLE = False
    print("[!] wikipedia-api not found — Wikipedia search disabled.")

# ── Optional: OpenAI LLM fallback ────────────────────────────────────────────
try:
    from openai import OpenAI
    _OAI_KEY = os.getenv("OPENAI_API_KEY", "")
    if _OAI_KEY:
        _OAI_CLIENT = OpenAI(api_key=_OAI_KEY)
        LLM_AVAILABLE = True
        print("[✔] OpenAI API loaded.")
    else:
        LLM_AVAILABLE = False
        print("[!] OPENAI_API_KEY not set — LLM fallback disabled.")
except ImportError:
    LLM_AVAILABLE = False
    print("[!] openai package not found — LLM fallback disabled.")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  EXPANDED BUILT-IN KNOWLEDGE BASE  (15+ topics, 100+ facts)
# ══════════════════════════════════════════════════════════════════════════════
BUILTIN_KNOWLEDGE = """
# ── AI & Machine Learning ────────────────────────────────────────────────────
Artificial intelligence is intelligence demonstrated by machines that mimics human cognitive functions such as learning and problem solving.
Machine learning enables systems to learn from data and improve over time without being explicitly reprogrammed.
Deep learning uses multi-layered neural networks to model complex patterns in large datasets such as images, audio, and text.
Reinforcement learning trains agents to make decisions by rewarding good actions and penalizing bad ones.
Supervised learning trains models on labeled input-output pairs so they can predict outputs for new inputs.
Unsupervised learning finds hidden patterns in data without labeled examples, using techniques like clustering and dimensionality reduction.
Transfer learning reuses a pre-trained model on a new but related task, saving time and computational resources.
A neural network is a system of interconnected nodes inspired by the human brain that processes data in layers.
Overfitting occurs when a model memorizes training data and fails to generalize to new data.
Regularization techniques such as dropout and L2 penalty reduce overfitting by constraining model complexity.
A confusion matrix shows the number of correct and incorrect predictions broken down by class.
The F1 score is the harmonic mean of precision and recall and is used for evaluating classification models.
Gradient descent is an optimization algorithm that minimizes a loss function by iteratively adjusting model weights.
Backpropagation computes gradients of the loss with respect to each weight using the chain rule for updating neural network parameters.
Batch normalization normalizes layer inputs during training to stabilize and speed up learning.

# ── NLP ───────────────────────────────────────────────────────────────────────
Natural language processing enables computers to understand, interpret, and generate human language.
Tokenization splits text into individual units called tokens such as words, subwords, or characters.
Lemmatization reduces words to their base dictionary form, for example running becomes run.
Stemming chops word suffixes to find a root form, which may not be a valid word.
Stopwords are common words like the, is, and a that are usually removed before NLP processing.
Named entity recognition identifies and classifies entities such as people, organizations, and locations in text.
Sentiment analysis determines the emotional tone of text as positive, negative, or neutral.
Part-of-speech tagging labels each word in a sentence with its grammatical role such as noun, verb, or adjective.
Coreference resolution determines which words in a text refer to the same entity.
Dependency parsing analyzes the grammatical structure of a sentence to find relationships between words.
Text summarization automatically produces a shorter version of a document while preserving key information.
Machine translation automatically converts text from one language to another using AI models.
Question answering is an NLP task where a model reads a passage and answers questions about it.

# ── Transformers & LLMs ───────────────────────────────────────────────────────
A transformer is a neural network architecture that uses self-attention to process entire sequences in parallel.
Self-attention allows a model to weigh the importance of every word relative to every other word in a sentence.
BERT is a bidirectional transformer pre-trained on masked language modeling and next sentence prediction tasks.
GPT is a generative pre-trained transformer that predicts the next word in a sequence to generate coherent text.
Large language models are transformer models trained on vast text corpora that can perform many NLP tasks without task-specific training.
Prompt engineering is the practice of designing inputs to guide large language models toward desired outputs.
Few-shot learning allows a model to perform a task given only a few examples provided in the prompt.
Zero-shot learning allows a model to handle tasks it was never explicitly trained on.
Fine-tuning adapts a pre-trained model to a specific task by continuing training on a smaller domain-specific dataset.
RAG stands for retrieval-augmented generation, a technique that enhances LLM responses by retrieving relevant documents first.

# ── Python ────────────────────────────────────────────────────────────────────
Python is a high-level interpreted programming language known for its readability and versatility.
Python was created by Guido van Rossum and first released in 1991.
Python supports multiple programming paradigms including procedural, object-oriented, and functional styles.
PEP 8 is the official Python style guide that provides conventions for writing clean, readable Python code.
List comprehensions in Python provide a concise way to create lists using a single line of code.
Decorators in Python are functions that modify the behavior of another function without changing its source code.
Generators in Python are functions that yield values one at a time using the yield keyword, saving memory.
Virtual environments isolate Python project dependencies to avoid version conflicts between projects.
pip is the standard package manager for Python used to install and manage third-party libraries.
Django and Flask are popular Python web frameworks used to build web applications and REST APIs.

# ── Data Science ──────────────────────────────────────────────────────────────
Data science is the field of extracting insights and knowledge from structured and unstructured data.
Pandas is a Python library for data manipulation and analysis using DataFrame and Series objects.
NumPy provides support for large multi-dimensional arrays and matrices along with mathematical functions.
Matplotlib and Seaborn are Python libraries used for creating static, animated, and interactive visualizations.
A DataFrame is a two-dimensional labeled data structure in Pandas similar to a spreadsheet or SQL table.
Feature engineering is the process of creating new input variables from raw data to improve model performance.
Data preprocessing includes steps like cleaning, normalizing, encoding, and splitting data before model training.
Cross-validation evaluates model performance by splitting data into multiple train/test folds to reduce variance.
Principal component analysis (PCA) reduces the dimensionality of data while preserving as much variance as possible.
K-means clustering partitions data into k groups where each point belongs to the cluster with the nearest mean.

# ── Computer Science ──────────────────────────────────────────────────────────
An algorithm is a step-by-step procedure for solving a problem or accomplishing a task.
A data structure is a way of organizing and storing data to enable efficient access and modification.
Time complexity describes how the runtime of an algorithm grows relative to the size of its input.
Big O notation expresses the upper bound of an algorithm's time or space complexity.
Object-oriented programming organizes code around objects that combine data and behavior into reusable units.
A class is a blueprint for creating objects that share common attributes and methods.
Inheritance allows a class to derive properties and methods from a parent class.
Polymorphism allows different classes to be treated as instances of the same parent class through shared interfaces.
An API (Application Programming Interface) defines how software components communicate with each other.
REST is an architectural style for designing web APIs using standard HTTP methods like GET, POST, PUT, DELETE.

# ── Web Development ───────────────────────────────────────────────────────────
HTML stands for HyperText Markup Language and is the standard language for creating web pages.
CSS stands for Cascading Style Sheets and is used to control the visual appearance of HTML elements.
JavaScript is a programming language that makes web pages interactive and dynamic in the browser.
React is a JavaScript library developed by Meta for building fast and interactive user interfaces.
Node.js is a JavaScript runtime that allows JavaScript to run on the server side outside the browser.
Flask is a lightweight Python web framework that makes it easy to build REST APIs and web applications.
FastAPI is a modern high-performance Python web framework for building APIs with automatic documentation.
A database is an organized collection of structured data stored electronically, typically accessed via SQL or NoSQL queries.
SQL stands for Structured Query Language and is used to manage and query relational databases.
MongoDB is a NoSQL document database that stores data in flexible JSON-like documents.

# ── Mathematics ───────────────────────────────────────────────────────────────
Linear algebra is the branch of mathematics dealing with vectors, matrices, and linear transformations.
A matrix is a rectangular array of numbers arranged in rows and columns used in linear algebra and ML.
A vector is a mathematical object with both magnitude and direction, represented as an array of numbers.
Probability is the measure of how likely an event is to occur, expressed as a number between 0 and 1.
Statistics is the science of collecting, analyzing, interpreting, and presenting data.
The mean is the average of a set of numbers calculated by dividing their sum by the count.
Standard deviation measures how spread out the values in a dataset are around the mean.
Calculus studies rates of change (derivatives) and accumulation (integrals) of quantities.
A derivative measures how a function changes as its input changes, used in optimization.
The Bayes theorem describes the probability of an event based on prior knowledge of related conditions.

# ── Cybersecurity ─────────────────────────────────────────────────────────────
Cybersecurity is the practice of protecting systems, networks, and programs from digital attacks.
Encryption converts readable data into an unreadable format to protect it from unauthorized access.
A firewall is a network security system that monitors and controls incoming and outgoing network traffic.
Phishing is a cyberattack that tricks users into revealing sensitive information through fake communications.
Two-factor authentication adds a second layer of security by requiring a second form of identity verification.
SQL injection is an attack that inserts malicious SQL code into a query to manipulate a database.
A VPN (Virtual Private Network) encrypts internet traffic and hides the user's IP address for privacy.
Hashing converts data into a fixed-size string of characters using a one-way function, used for passwords.

# ── Cloud Computing ───────────────────────────────────────────────────────────
Cloud computing delivers computing services like servers, storage, and databases over the internet.
AWS (Amazon Web Services) is the world's most widely used cloud platform offering hundreds of services.
Docker is a platform that packages applications into containers for consistent deployment across environments.
Kubernetes is an open-source system for automating the deployment and scaling of containerized applications.
Serverless computing runs code in response to events without managing servers, billed per execution.
CI/CD stands for Continuous Integration and Continuous Deployment, automating software build and release pipelines.

# ── Science ───────────────────────────────────────────────────────────────────
Physics is the natural science that studies matter, energy, space, time, and their interactions.
Chemistry is the science of substances, their properties, structure, and the reactions they undergo.
Biology is the science of life and living organisms including their structure, function, and evolution.
DNA stands for deoxyribonucleic acid and carries the genetic instructions for all living organisms.
The theory of evolution by natural selection was proposed by Charles Darwin in 1859.
The speed of light in a vacuum is approximately 299,792 kilometers per second.
Quantum mechanics describes the behavior of matter and energy at the smallest scales of atoms and subatomic particles.
The Big Bang theory states that the universe originated from an extremely hot and dense state about 13.8 billion years ago.
Photosynthesis is the process by which green plants convert sunlight, water, and carbon dioxide into glucose and oxygen.
Newton's three laws of motion describe the relationship between the motion of an object and the forces acting on it.

# ── History & General Knowledge ───────────────────────────────────────────────
World War II was a global conflict from 1939 to 1945 involving most of the world's nations.
The French Revolution began in 1789 and fundamentally transformed France's political and social structure.
The Industrial Revolution started in Britain in the late 18th century and transformed manufacturing and society.
The internet was invented based on ARPANET research in the late 1960s and became public in the early 1990s.
The United Nations was founded in 1945 to promote international peace, security, and cooperation.
Democracy is a system of government where citizens exercise power by voting for their representatives.
The Renaissance was a cultural movement in Europe from the 14th to 17th centuries that revived classical art and learning.
"""


def load_knowledge(filepath: str = "docs.txt") -> str:
    path = Path(filepath)
    if path.exists():
        extra = path.read_text(encoding="utf-8").strip()
        print(f"[✔] Loaded '{filepath}' ({len(extra)} chars of extra knowledge).")
        return BUILTIN_KNOWLEDGE + "\n" + extra
    else:
        path.write_text(
            "# Add your own facts here, one sentence per line.\n"
            "# Example: The Eiffel Tower is located in Paris, France.\n",
            encoding="utf-8"
        )
        print(f"[✔] Created empty '{filepath}'. Add your own facts there!")
        return BUILTIN_KNOWLEDGE


# ══════════════════════════════════════════════════════════════════════════════
# 2.  INTENTS
# ══════════════════════════════════════════════════════════════════════════════
INTENTS = {
    "greeting":     {"patterns": ["hello","hi","hey","good morning","good afternoon","good evening","howdy","what's up","greetings","sup"],"responses": ["Hello! Ask me anything — I know about AI, science, history, coding, and much more!","Hi there! I have a broad knowledge base. What would you like to know?","Hey! Feel free to ask me anything."]},
    "farewell":     {"patterns": ["bye","goodbye","see you","take care","exit","quit","later","farewell","cya"],"responses": ["Goodbye! Come back anytime.","See you later! Take care.","Bye! It was great chatting with you."]},
    "thanks":       {"patterns": ["thanks","thank you","appreciate it","thx","ty","much appreciated"],"responses": ["You're welcome!","Happy to help!","Anytime!"]},
    "name":         {"patterns": ["what is your name","who are you","what are you called","your name","introduce yourself"],"responses": ["I'm NLPBot — an intelligent chatbot with knowledge across AI, science, history, coding, math, and more!"]},
    "capabilities": {"patterns": ["what can you do","help","capabilities","what do you know","your abilities","topics"],"responses": ["I can answer questions on: AI/ML, NLP, Python, Data Science, Web Dev, Math, Physics, Chemistry, Biology, History, Cybersecurity, Cloud Computing, and more. I can also search Wikipedia for anything I don't know locally. Just ask!"]},
    "age":          {"patterns": ["how old are you","your age","when were you created"],"responses": ["I was just created — forever young and always learning!"]},
    "creator":      {"patterns": ["who made you","who created you","who built you","who developed you"],"responses": ["I was built by a developer using Python, NLTK, Sentence-Transformers, and the Wikipedia API!"]},
    "joke":         {"patterns": ["tell me a joke","joke","make me laugh","funny"],"responses": ["Why do Python programmers prefer dark mode? Because light attracts bugs! 🐛","Why did the ML model break up with the dataset? It said there was no connection! 😄","I asked my AI to write me a poem. It said it was still training on rhymes."]},
    "motivation":   {"patterns": ["motivate me","inspire me","motivation","encourage me","give me a quote"],"responses": ['"The only way to do great work is to love what you do." — Steve Jobs','"In the middle of every difficulty lies opportunity." — Albert Einstein','"It does not matter how slowly you go as long as you do not stop." — Confucius']},
    "time":         {"patterns": ["what time is it","current time","what's the time","time now"],"responses": ["__DYNAMIC_TIME__"]},
    "date":         {"patterns": ["what is today's date","today's date","what day is it","current date"],"responses": ["__DYNAMIC_DATE__"]},
    "math":         {"patterns": ["calculate","compute","what is 2","what is 3","solve","math problem","arithmetic"],"responses": ["__DYNAMIC_MATH__"]},
    "wikipedia":    {"patterns": ["search wikipedia","wiki","wikipedia","look up","search for","find info on","tell me about","who is","what is the history of"],"responses": ["__DYNAMIC_WIKI__"]},
    "history_show": {"patterns": ["show history","chat history","previous messages","what did i say","our conversation"],"responses": ["__DYNAMIC_HISTORY__"]},
    "history_clear":{"patterns": ["clear history","delete history","reset history","forget our conversation"],"responses": ["__DYNAMIC_CLEAR_HISTORY__"]},
    "teach":        {"patterns": ["learn that","remember that","add to knowledge","teach you"],"responses": ["__DYNAMIC_TEACH__"]},
    "sentiment":    {"patterns": ["detect my mood","how am i feeling","analyze my sentiment","what's my emotion"],"responses": ["__DYNAMIC_SENTIMENT__"]},
    "feedback_good":{"patterns": ["good bot","great answer","correct","perfect","that's right","awesome","excellent","well done"],"responses": ["Thank you! 😊 Glad that was helpful.","Great to hear! I'm always improving."]},
    "feedback_bad": {"patterns": ["wrong","incorrect","bad answer","that's wrong","not right","you're wrong","bad bot"],"responses": ["I'm sorry about that! I'm still learning. Could you tell me the correct answer?","Thanks for the correction — I'll try to do better!"]},
    "weather":      {"patterns": ["weather","temperature","forecast","raining","sunny"],"responses": ["I don't have live weather access, but check weather.com or your phone's assistant for real-time info!"]},
    "summarize":    {"patterns": ["summarize","summary","tldr","brief me","give me a summary"],"responses": ["__DYNAMIC_SUMMARIZE__"]},
}


# ══════════════════════════════════════════════════════════════════════════════
# 3.  NLP UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words("english"))


def preprocess(text: str) -> str:
    tokens = word_tokenize(text.lower())
    return " ".join(
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words and t not in string.punctuation
    )


def detect_intent(text: str) -> str | None:
    lowered = text.lower()
    for intent, data in INTENTS.items():
        for p in data["patterns"]:
            if p in lowered:
                return intent
    return None


def tfidf_response(user_input: str, sentences: list[str]) -> tuple[str | None, float]:
    processed = [preprocess(s) for s in sentences]
    pi = preprocess(user_input)
    if not pi.strip():
        return None, 0.0
    vec  = TfidfVectorizer()
    mat  = vec.fit_transform(processed + [pi])
    sims = cosine_similarity(mat[-1], mat[:-1])
    idx  = int(np.argmax(sims))
    score = float(sims[0, idx])
    return (sentences[idx] if score >= 0.1 else None), score


def semantic_response(user_input: str, sentences: list[str], embeddings) -> tuple[str | None, float]:
    qe   = _ST_MODEL.encode(user_input, convert_to_tensor=True)
    sims = st_util.cos_sim(qe, embeddings)[0].cpu().numpy()
    idx  = int(np.argmax(sims))
    score = float(sims[idx])
    return (sentences[idx] if score >= 0.25 else None), score


def safe_math(expr: str) -> str | None:
    expr = re.sub(r"[^0-9+\-*/().\s]", "", expr)
    try:
        return str(eval(expr, {"__builtins__": {}}))     # noqa: S307
    except Exception:
        return None


POSITIVE_WORDS = {"happy","great","good","love","excellent","wonderful","amazing","fantastic","joy","excited","cheerful","delighted","positive"}
NEGATIVE_WORDS = {"sad","bad","hate","terrible","awful","horrible","angry","depressed","upset","frustrated","anxious","miserable","negative"}

def simple_sentiment(text: str) -> str:
    tokens = set(word_tokenize(text.lower()))
    pos = len(tokens & POSITIVE_WORDS)
    neg = len(tokens & NEGATIVE_WORDS)
    if pos > neg:  return "positive 😊"
    if neg > pos:  return "negative 😔"
    return "neutral 😐"


def extractive_summarize(text: str, n: int = 3) -> str:
    sentences = sent_tokenize(text)
    if len(sentences) <= n:
        return text
    if SEMANTIC_AVAILABLE:
        embeddings = _ST_MODEL.encode(sentences, convert_to_tensor=True)
        doc_emb    = _ST_MODEL.encode(text, convert_to_tensor=True)
        scores     = st_util.cos_sim(doc_emb, embeddings)[0].cpu().numpy()
    else:
        vec    = TfidfVectorizer()
        mat    = vec.fit_transform([preprocess(s) for s in sentences])
        doc_v  = TfidfVectorizer().fit_transform([preprocess(text)])
        scores = np.array([float(cosine_similarity(doc_v, mat[i])) for i in range(len(sentences))])
    top_idx = sorted(np.argsort(scores)[-n:])
    return " ".join(sentences[i] for i in top_idx)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  WIKIPEDIA SEARCH
# ══════════════════════════════════════════════════════════════════════════════
_wiki_cache: dict[str, str] = {}

def wiki_search(query: str) -> str | None:
    if not WIKI_AVAILABLE:
        return None
    # Extract a clean topic from the query
    topic = re.sub(
        r"(search wikipedia|wiki|wikipedia|look up|search for|find info on|tell me about|who is|what is the history of)",
        "", query, flags=re.IGNORECASE
    ).strip().strip("?").strip()

    if not topic:
        return None

    if topic.lower() in _wiki_cache:
        return _wiki_cache[topic.lower()]

    print(f"[Wiki] Searching Wikipedia for: '{topic}' …", end=" ", flush=True)
    page = _WIKI.page(topic)
    if page.exists():
        summary = page.summary[:800]   # First 800 chars
        _wiki_cache[topic.lower()] = summary
        print("found!")
        return summary
    print("not found.")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 5.  LLM FALLBACK  (OpenAI)
# ══════════════════════════════════════════════════════════════════════════════
def llm_response(user_input: str) -> str | None:
    if not LLM_AVAILABLE:
        return None
    try:
        print("[LLM] Querying OpenAI …", end=" ", flush=True)
        resp = _OAI_CLIENT.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful, concise assistant. Answer in 2-3 sentences."},
                {"role": "user",   "content": user_input},
            ],
            max_tokens=150,
            temperature=0.7,
        )
        print("done.")
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"failed ({e}).")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 6.  CHAT HISTORY
# ══════════════════════════════════════════════════════════════════════════════
HISTORY_FILE = "chat_history.json"

def load_history() -> list[dict]:
    if Path(HISTORY_FILE).exists():
        try:
            return json.loads(Path(HISTORY_FILE).read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def save_history(history: list[dict]) -> None:
    Path(HISTORY_FILE).write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )

def format_history(history: list[dict], n: int = 10) -> str:
    if not history:
        return "No chat history yet."
    lines = []
    for e in history[-n:]:
        lines += [f"[{e['time']}] You: {e['user']}", f"[{e['time']}] Bot: {e['bot']}"]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  CHATBOT CLASS
# ══════════════════════════════════════════════════════════════════════════════
class NLPChatbot:
    def __init__(self):
        self.raw_knowledge = load_knowledge("docs.txt")
        # Filter out comment lines
        clean = "\n".join(
            l for l in self.raw_knowledge.splitlines()
            if not l.strip().startswith("#") and l.strip()
        )
        self.sentences = sent_tokenize(clean)

        if SEMANTIC_AVAILABLE:
            print(f"[✔] Computing embeddings for {len(self.sentences)} sentences …", end=" ")
            self.embeddings = _ST_MODEL.encode(self.sentences, convert_to_tensor=True)
            print("done.")
        else:
            self.embeddings = None

        self.history: list[dict] = load_history()
        self._pending_summarize: bool = False
        print(f"[✔] Ready! Loaded {len(self.history)} history messages.\n")

    # ── dynamic resolvers ────────────────────────────────────────────────────
    def _dynamic(self, intent: str, user_input: str) -> str:
        now = datetime.now()

        if intent == "time":
            return f"The current time is {now.strftime('%I:%M %p')}."

        if intent == "date":
            return f"Today is {now.strftime('%A, %B %d, %Y')}."

        if intent == "math":
            expr = re.sub(r"[a-zA-Z\?,!]", " ", user_input)
            result = safe_math(expr)
            return f"Result: {result}" if result else \
                   "Please give me a math expression, e.g. 'calculate 25 * 4'."

        if intent == "wikipedia":
            result = wiki_search(user_input)
            if result:
                # Add to live knowledge base
                new_sentences = sent_tokenize(result)
                self.sentences.extend(new_sentences)
                if SEMANTIC_AVAILABLE:
                    new_emb = _ST_MODEL.encode(new_sentences, convert_to_tensor=True)
                    import torch
                    self.embeddings = torch.cat([self.embeddings, new_emb], dim=0)
                return result
            return "I couldn't find a Wikipedia article on that. Try rephrasing the topic."

        if intent == "history_show":
            return "── Last 10 Messages ──\n" + format_history(self.history)

        if intent == "history_clear":
            self.history.clear()
            save_history(self.history)
            return "History cleared! Starting fresh. 🗑️"

        if intent == "teach":
            m = re.search(
                r"(?:learn that|remember that|add to knowledge|teach you)[:\s]+(.+)",
                user_input, re.IGNORECASE
            )
            if m:
                fact = m.group(1).strip()
                self.sentences.append(fact)
                if SEMANTIC_AVAILABLE:
                    import torch
                    new_emb = _ST_MODEL.encode([fact], convert_to_tensor=True)
                    self.embeddings = torch.cat([self.embeddings, new_emb], dim=0)
                return f"Got it! I've learned: \"{fact}\""
            return "Tell me what to learn, e.g.: 'Learn that water boils at 100°C.'"

        if intent == "sentiment":
            text = self.history[-1]["user"] if self.history else user_input
            return f"Your sentiment seems to be: {simple_sentiment(text)}"

        if intent == "summarize":
            self._pending_summarize = True
            return "Sure! Please paste the text you want me to summarize."

        return "I'm working on that feature!"

    # ── main pipeline ────────────────────────────────────────────────────────
    def get_response(self, user_input: str) -> str:
        user_input = user_input.strip()
        if not user_input:
            return "Please say something — I'm listening!"

        # Summarize mode — next message is the text to summarize
        if self._pending_summarize:
            self._pending_summarize = False
            summary = extractive_summarize(user_input)
            response = f"📝 Summary:\n{summary}"
            self._log(user_input, response)
            return response

        # 1. Intent detection
        intent = detect_intent(user_input)
        if intent:
            tmpl = random.choice(INTENTS[intent]["responses"])
            response = self._dynamic(intent, user_input) if tmpl.startswith("__DYNAMIC_") else tmpl
            self._log(user_input, response)
            return response

        # 2. Semantic / TF-IDF knowledge search
        if SEMANTIC_AVAILABLE:
            answer, score = semantic_response(user_input, self.sentences, self.embeddings)
        else:
            answer, score = tfidf_response(user_input, self.sentences)

        if answer:
            self._log(user_input, answer)
            return answer

        # 3. Wikipedia live search (auto, no need for "search wikipedia" prefix)
        if WIKI_AVAILABLE:
            wiki_ans = wiki_search(user_input)
            if wiki_ans:
                new_sentences = sent_tokenize(wiki_ans)
                self.sentences.extend(new_sentences)
                if SEMANTIC_AVAILABLE:
                    import torch
                    new_emb = _ST_MODEL.encode(new_sentences, convert_to_tensor=True)
                    self.embeddings = torch.cat([self.embeddings, new_emb], dim=0)
                self._log(user_input, wiki_ans)
                return f"[Wikipedia] {wiki_ans}"

        # 4. LLM fallback
        if LLM_AVAILABLE:
            llm_ans = llm_response(user_input)
            if llm_ans:
                self._log(user_input, llm_ans)
                return f"[AI] {llm_ans}"

        # 5. Final fallback
        response = (
            "I don't have information on that yet. You can:\n"
            "  • Say 'Wikipedia <topic>' to fetch a live Wikipedia summary\n"
            "  • Say 'Learn that <fact>' to teach me directly\n"
            "  • Add facts to docs.txt and restart"
        )
        self._log(user_input, response)
        return response

    def _log(self, user: str, bot: str) -> None:
        self.history.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": user, "bot": bot
        })
        save_history(self.history)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "═" * 65)
    print("  NLPBot — Unlimited Knowledge Edition")
    print("  ✔ 100+ built-in facts across 15 topics")
    print("  ✔ Live Wikipedia search for any unknown topic")
    print("  ✔ OpenAI LLM fallback (set OPENAI_API_KEY)")
    print("  ✔ Teach me | Summarize | History | Math | Sentiment")
    print("  Type 'bye' to exit")
    print("═" * 65 + "\n")

    bot = NLPChatbot()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye! Chat saved.")
            break

        if not user_input:
            continue

        response = bot.get_response(user_input)
        print(f"Bot: {response}\n")

        if detect_intent(user_input) == "farewell":
            break


if __name__ == "__main__":
    main()