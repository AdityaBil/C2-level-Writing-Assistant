# ✍️ C2 English Writing Assistant

An elite, high-precision English writing and vocabulary assistant powered by modern LLM APIs (**Groq**, **OpenAI**, **OpenRouter**, or custom OpenAI-compatible endpoints).

Designed specifically for writers, researchers, students, and professionals who want **C1/C2 proficiency**, syntactic elegance, and deep lexical nuance without running heavy local models.

---

## ⚡ Key Highlights

- **Zero Local GPU Requirement**: Inference is performed via high-speed external LLM APIs (e.g. Groq running Llama-3.3-70B in milliseconds).
- **Extremely Lightweight**: No PyTorch, CUDA, BitsAndBytes, or multi-gigabyte model downloads needed on your machine.
- **Context-Sensitive Vocabulary**: Understands that synonyms are not universally interchangeable; explains subtle connotations, registers, and collocations.
- **Three Specialized Modes**:
  1. **✨ Enhance Phrase**: Elevates ordinary sentences into sophisticated C1/C2 English.
  2. **📚 C2 Synonyms**: Discovers nuanced alternatives with precise definitions, examples, and interchangeability warnings.
  3. **🖋️ Polish (Preserve Voice)**: Polishes prose while strictly preserving the author's authentic personal voice.
- **7 Stylistic Presets**: `Natural C2` (default), `Academic`, `Literary`, `Philosophical`, `Poetic`, `Formal`, and `Concise`.

---

## 💻 Hardware Requirements

> **No dedicated GPU or local model download is required.**

The application runs smoothly on any low-spec laptop or desktop. All computational heavy lifting happens remotely on the LLM API provider's infrastructure.

---

## 🚀 Quick Start Guide

### 1. Clone or Open the Repository

```bash
cd llama3-c2-writing-assistant
```

### 2. Create and Activate a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Lightweight Dependencies

```bash
pip install -r requirements.txt
```

*(This takes only a few seconds as it installs only Streamlit, python-dotenv, and lightweight API SDKs).*

---

## 🔑 API Configuration

### Step 1: Create your `.env` file

Copy `.env.example` to `.env`:

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

**macOS / Linux / Git Bash:**
```bash
cp .env.example .env
```

### Step 2: Add your API Key

Open `.env` in any text editor and add your key:

```env
# Primary LLM Provider: 'groq', 'openai', 'openrouter', or 'custom'
LLM_PROVIDER=groq

# Your API Key
LLM_API_KEY=gsk_your_groq_api_key_here

# Model name
LLM_MODEL=llama-3.3-70b-versatile
```

#### Where to get API keys:
- **Groq (Recommended - Free & Blazing Fast)**: [console.groq.com/keys](https://console.groq.com/keys)
- **OpenAI**: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **OpenRouter**: [openrouter.ai/keys](https://openrouter.ai/keys)

*(Note: You can also enter or change your API key directly in the Streamlit sidebar at any time).*

---

## 🖥️ Running the Application

Launch the Streamlit web interface:

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 🎯 Application Modes & Capabilities

```
╔══════════════════════════════════════════════╗
║        C2 ENGLISH WRITING ASSISTANT         ║
║      Precision • Elegance • Vocabulary      ║
╠══════════════════════════════════════════════╣
║  Mode: [Enhance] [Synonyms] [Polish]         ║
║  Style: [Natural C2 ▼]                       ║
╚══════════════════════════════════════════════╝
```

### 1. ✨ Enhance Phrase
Rewrites ordinary prose into sophisticated C1/C2 English while preserving meaning, intent, factual integrity, and emotional register.
- **Example Input:** `"This is an ideal opportunity."`
- **Output:** `"This constitutes an exceptionally auspicious opportunity."`
- **Lexical Insights:** Explains specific lexical choices (e.g. *auspicious* = conveying favorable prospects for future success).

### 2. 📚 C2 Synonyms
Provides 5–8 high-caliber alternatives with explicit semantic distinctions, registers, and examples.
- **Example Input:** `"ideal"`
- **Structured Cards:**
  - **OPTIMAL**: Best or most favorable under particular conditions.
  - **EXEMPLARY**: Serving as an outstanding model or benchmark.
  - **QUINTESSENTIAL**: Representing the purest, most characteristic embodiment.
  - **CONSUMMATE**: Demonstrating the highest degree of mastery or perfection.
  - **AUSPICIOUS**: Indicating favorable circumstances or future success.
- **Contextual Interchangeability Note:** Clarifies why these words cannot be swapped universally.

### 3. 🖋️ Polish (Preserve Voice)
Refines flow, rhythm, and clarity without turning the author's style into generic AI prose.
- **Example Input:** `"Even though I disagree with this philosophy, studying it taught me how to embrace boredom."`
- **Output:** `"Although I fundamentally disagree with this philosophy, engaging with it nevertheless taught me the discipline of embracing boredom."`

---

## 📁 Project Structure

```text
llama3-c2-writing-assistant/
│
├── app.py                  # Lightweight Streamlit frontend
├── llm_client.py           # Provider-agnostic API client (Groq, OpenAI, Custom)
├── prompts.py              # Elite C1/C2 system instructions, modes & style presets
├── requirements.txt        # Lightweight dependencies (no PyTorch / GPU required)
├── .env.example            # Environment configuration template
├── .gitignore              # Ignores .env, .venv, caches, and weight files
├── README.md               # Documentation & setup guide
│
└── training/               # Optional legacy QLoRA fine-tuning pipeline
    ├── README.md           # Instructions for offline fine-tuning
    ├── requirements.txt    # Heavy ML dependencies (PyTorch, Transformers, PEFT)
    ├── train.py            # QLoRA fine-tuning script
    ├── prep_data.py        # Dataset preprocessing
    ├── inference.py        # Local CLI inference
    ├── c2_engine.py        # Local 4-bit model engine
    ├── data/               # Seed dataset
    └── llama-3-c2-adapter/ # LoRA checkpoints
```

---

## ☁️ Deployment Guide

The application is 100% cloud-ready and can be deployed with zero GPU requirements on any of the following platforms:

### 1. Streamlit Community Cloud (Recommended & Free)
1. Push this repository to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and click **New App**.
3. Select your repository, branch (`main`), and set **Main file path** to `app.py`.
4. In **Advanced Settings -> Secrets**, paste your API credentials:
   ```toml
   LLM_PROVIDER = "groq"
   LLM_API_KEY = "gsk_your_groq_api_key"
   LLM_MODEL = "llama-3.3-70b-versatile"
   ```
5. Click **Deploy**!

### 2. Hugging Face Spaces
1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces) selecting the **Streamlit** SDK.
2. Push your files or connect your GitHub repository.
3. In **Settings -> Variables and Secrets**, add `LLM_API_KEY` (and optionally `LLM_PROVIDER`, `LLM_MODEL`).

### 3. Docker / Render / Railway / Google Cloud Run
Build and run the container locally or on any cloud container service:
```bash
docker build -t c2-writing-assistant .
docker run -p 8501:8501 -e LLM_API_KEY=your_key c2-writing-assistant
```

---

## 🛡️ Security & Privacy

- `.env` is included in `.gitignore` to prevent committing sensitive API keys.
- Never hardcode API keys into code files.
- The modular client supports on-premise / local OpenAI-compatible endpoints (e.g., Ollama or vLLM) if you require complete data isolation.
