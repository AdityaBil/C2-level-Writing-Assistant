"""
app.py
======

Streamlit user interface for the Elite C1/C2 English Writing Assistant.
Powered by external LLM APIs (Groq, OpenAI, OpenRouter, Custom).

Zero local GPU or heavy ML dependencies required.
"""

from __future__ import annotations

import os
import streamlit as st

import llm_client
from llm_client import (
    AuthenticationError,
    GenerationResult,
    LLMClient,
    LLMClientError,
    MissingAPIKeyError,
    ModelNotFoundError,
    NetworkTimeoutError,
    RateLimitError,
)
import prompts
from prompts import AVAILABLE_STYLES, DEFAULT_STYLE

# --------------------------------------------------------------------------- #
# Page Configuration
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="C2 English Writing Assistant",
    page_icon="✍️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Custom Styling (Modern, Clean, Responsive)
# --------------------------------------------------------------------------- #

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

.block-container {
    padding-top: 1.8rem;
    padding-bottom: 3.5rem;
    max-width: 860px;
}

.main-header {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.3rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin-bottom: 0.15rem;
}

.sub-header {
    font-size: 0.95rem;
    color: #818cf8;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}

.c2-card {
    background: rgba(99, 102, 241, 0.06);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-left: 5px solid #6366f1;
    border-radius: 12px;
    padding: 1.3rem 1.45rem;
    font-size: 1.15rem;
    line-height: 1.75;
    margin: 1rem 0;
    box-shadow: 0 4px 20px -2px rgba(99, 102, 241, 0.08);
}

.syn-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(128, 128, 128, 0.22);
    border-radius: 10px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 0.85rem;
    transition: all 0.2s ease;
}

.syn-card:hover {
    border-color: rgba(99, 102, 241, 0.6);
    background: rgba(99, 102, 241, 0.04);
}

.syn-term {
    font-weight: 800;
    letter-spacing: 0.05em;
    font-size: 1.15rem;
    color: #6366f1;
    display: inline-block;
    margin-bottom: 0.3rem;
}

.syn-badge {
    background: rgba(99, 102, 241, 0.15);
    color: #818cf8;
    border-radius: 4px;
    padding: 0.15rem 0.45rem;
    font-size: 0.75rem;
    font-weight: 700;
    margin-left: 0.5rem;
    text-transform: uppercase;
}

.syn-example {
    background: rgba(128, 128, 128, 0.08);
    border-left: 3px solid rgba(99, 102, 241, 0.7);
    padding: 0.5rem 0.85rem;
    border-radius: 6px;
    font-style: italic;
    margin-top: 0.55rem;
    font-size: 0.96rem;
    line-height: 1.55;
}

.note-box {
    background: rgba(245, 158, 11, 0.07);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-left: 4px solid #f59e0b;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    font-size: 0.95rem;
    line-height: 1.6;
}

.insight-box {
    background: rgba(16, 185, 129, 0.06);
    border: 1px solid rgba(16, 185, 129, 0.25);
    border-left: 4px solid #10b981;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    font-size: 0.95rem;
    line-height: 1.6;
}

.stats-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(128, 128, 128, 0.1);
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.8rem;
    font-weight: 500;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
}

.chip-btn {
    font-size: 0.8rem !important;
    padding: 0.25rem 0.5rem !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Sidebar: Provider & Model Configuration
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.markdown("### ⚡ LLM API Engine")

    # Provider Selection
    env_provider = llm_client.get_config_val("LLM_PROVIDER", "groq").lower()
    provider_options = ["Groq (Ultra Fast)", "OpenAI", "OpenRouter", "Custom Endpoint", "🧪 Offline Demo Mode"]
    provider_mapping = {
        "Groq (Ultra Fast)": "groq",
        "OpenAI": "openai",
        "OpenRouter": "openrouter",
        "Custom Endpoint": "custom",
        "🧪 Offline Demo Mode": "demo",
    }

    # Determine default provider index
    default_idx = 0
    if env_provider == "openai":
        default_idx = 1
    elif env_provider == "openrouter":
        default_idx = 2
    elif env_provider == "custom":
        default_idx = 3
    elif env_provider == "demo":
        default_idx = 4

    selected_provider_label = st.selectbox(
        "API Provider",
        options=provider_options,
        index=default_idx,
        help="Select the external LLM provider. Groq offers ultra-low latency inference with Llama 3 models."
    )
    provider_key = provider_mapping[selected_provider_label]

    # Model Selection / Input
    default_model = llm_client.get_config_val("LLM_MODEL", llm_client.DEFAULT_MODELS.get(provider_key, "qwen/qwen3.8-27b"))

    if provider_key == "groq":
        model_options = ["qwen/qwen3.8-27b", "groq/compound-mini", "groq/compound", "openai/gpt-oss-20b", "openai/gpt-oss-120b", "Custom Groq Model..."]
        model_choice = st.selectbox("Model", options=model_options, index=0)
        if model_choice == "Custom Groq Model...":
            active_model = st.text_input("Custom Model ID", value="qwen/qwen3.8-27b")
        else:
            active_model = model_choice
    elif provider_key == "openai":
        model_options = ["gpt-4o-mini", "gpt-4o", "Custom OpenAI Model..."]
        model_choice = st.selectbox("Model", options=model_options, index=0)
        if model_choice == "Custom OpenAI Model...":
            active_model = st.text_input("Custom Model ID", value="gpt-4o-mini")
        else:
            active_model = model_choice
    elif provider_key == "openrouter":
        active_model = st.text_input("OpenRouter Model", value=llm_client.get_config_val("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct"))
    elif provider_key == "demo":
        active_model = "c2-demo-engine"
        st.caption("🧪 Demo engine returns realistic C2 samples instantly.")
    else:
        active_model = st.text_input("Custom Model ID", value=default_model)

    # Custom Base URL (if applicable)
    custom_base_url = None
    if provider_key == "custom":
        custom_base_url = st.text_input(
            "API Base URL",
            value=llm_client.get_config_val("LLM_BASE_URL", "http://localhost:11434/v1"),
            help="E.g., http://localhost:11434/v1 for Ollama, https://api.together.xyz/v1, etc."
        )

    # API Key Handling (Env Var vs Manual Input)
    env_api_key = llm_client.get_config_val("LLM_API_KEY", "")
    if provider_key != "demo":
        api_key_input = st.text_input(
            f"{provider_key.upper()} API Key",
            value=env_api_key,
            type="password",
            help="Reads from .env / secrets by default. You can also paste your key directly here.",
        )
    else:
        api_key_input = "demo-mode"

    # Test Connection Button
    col_conn1, col_conn2 = st.columns([1, 1])
    with col_conn1:
        if st.button("🔌 Test API", use_container_width=True):
            if not api_key_input:
                st.sidebar.error("Please provide an API key.")
            else:
                test_client = LLMClient(
                    provider=provider_key,
                    api_key=api_key_input,
                    model=active_model,
                    base_url=custom_base_url,
                    timeout=10.0,
                )
                with st.spinner("Testing..."):
                    ok, msg = test_client.test_connection()
                if ok:
                    st.sidebar.success(msg)
                else:
                    st.sidebar.error(f"Failed: {msg}")

    with col_conn2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    st.divider()

    # Generation Parameters
    st.markdown("### 🎛️ Hyperparameters")
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.5,
        value=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        step=0.05,
        help="Lower values yield more focused, deterministic phrasing; higher values encourage creative flair."
    )
    max_tokens = st.slider(
        "Max Output Tokens",
        min_value=128,
        max_value=2048,
        value=int(os.getenv("LLM_MAX_TOKENS", "768")),
        step=64,
        help="Maximum length of the generated response."
    )

    st.divider()
    st.markdown(
        """
        <div style='font-size: 0.8rem; opacity: 0.7; line-height: 1.4;'>
        <b>🔒 Lightweight Architecture</b><br>
        • Zero local GPU requirement<br>
        • Inference via high-speed API<br>
        • Context-aware C1/C2 mastery
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

st.markdown('<div class="main-header">✍️ C2 English Writing Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Precision • Elegance • Nuance • Vocabulary</div>', unsafe_allow_html=True)

# Check API Key Readiness
if not api_key_input:
    st.info(
        "👋 **Welcome!** To get started, provide your API key in the sidebar or create a `.env` file from `.env.example`.\n\n"
        "💡 *Tip: Groq offers free, ultra-fast API keys at [console.groq.com](https://console.groq.com/keys).*"
    )


# --------------------------------------------------------------------------- #
# Generation Execution Helper
# --------------------------------------------------------------------------- #

def execute_llm_request(messages: list) -> GenerationResult | None:
    """Execute LLM request with structured error handling and visual feedback."""
    if not api_key_input:
        st.error("⚠️ **Missing API Key**: Please enter your API key in the sidebar or configure `.env`.")
        return None

    client = LLMClient(
        provider=provider_key,
        api_key=api_key_input,
        model=active_model,
        base_url=custom_base_url,
    )

    try:
        with st.spinner("Generating with C2 precision…"):
            result = client.generate(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return result

    except MissingAPIKeyError as exc:
        st.error(f"🔑 **Missing API Key**: {exc}")
    except AuthenticationError as exc:
        st.error(f"🚫 **Authentication Error**: {exc}")
    except RateLimitError as exc:
        st.warning(f"⏳ **Rate Limit Hit**: {exc}")
    except ModelNotFoundError as exc:
        st.error(f"🔍 **Model Not Found**: {exc}")
    except NetworkTimeoutError as exc:
        st.error(f"⏱️ **Timeout**: {exc}")
    except LLMClientError as exc:
        st.error(f"❌ **API Failure**: {exc}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unexpected error occurred: {type(exc).__name__}: {exc}")

    return None


# --------------------------------------------------------------------------- #
# Mode Tabs
# --------------------------------------------------------------------------- #

tab_enhance, tab_synonyms, tab_polish, tab_about = st.tabs([
    "✨ Enhance Phrase",
    "📚 C2 Synonyms",
    "🖋️ Polish Voice",
    "ℹ️ Architecture & Guide",
])


# =========================================================================== #
# TAB 1: ENHANCE PHRASE
# =========================================================================== #

with tab_enhance:
    st.markdown("#### Elevate phrases and sentences into sophisticated C1/C2 English")
    st.caption("Rewrites ordinary prose for precision, elegance, and natural educated native register.")

    # Style selector & Sample Prompts
    col_style, col_chips = st.columns([1, 2])
    with col_style:
        selected_style_enh = st.selectbox(
            "Style Register",
            options=AVAILABLE_STYLES,
            index=0,
            key="enhance_style_select",
        )

    with col_chips:
        st.markdown("<div style='font-size: 0.8rem; font-weight: 600; opacity: 0.8; margin-bottom: 0.2rem;'>💡 Sample Prompts:</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        if c1.button("Opportunity", key="enh_chip1", use_container_width=True):
            st.session_state["enhance_text_area"] = "This is an ideal opportunity."
            st.rerun()
        if c2.button("Complexity", key="enh_chip2", use_container_width=True):
            st.session_state["enhance_text_area"] = "I don't understand this problem."
            st.rerun()
        if c3.button("Strategy Pivot", key="enh_chip3", use_container_width=True):
            st.session_state["enhance_text_area"] = "We need to change how we work because the old way is too slow and hard to maintain."
            st.rerun()

    enhance_input = st.text_area(
        "Enter sentence or passage:",
        height=120,
        placeholder="E.g., This is an ideal opportunity.",
        key="enhance_text_area",
    )

    # Input Metrics
    char_count = len(enhance_input)
    word_count = len(enhance_input.split()) if enhance_input.strip() else 0
    st.markdown(
        f"<div style='font-size: 0.75rem; opacity: 0.65; margin-top: -0.5rem; margin-bottom: 0.8rem;'>"
        f"Words: {word_count} | Characters: {char_count}</div>",
        unsafe_allow_html=True,
    )

    col_btn1, col_btn2, _ = st.columns([1.2, 1, 2])
    with col_btn1:
        generate_enh = st.button("✨ Enhance to C2", type="primary", use_container_width=True, key="btn_enhance")
    with col_btn2:
        if st.button("🗑️ Clear", use_container_width=True, key="btn_enh_clear"):
            st.session_state["enhance_text_area"] = ""
            st.session_state["enhance_result"] = None
            st.rerun()

    if generate_enh:
        if not enhance_input.strip():
            st.warning("Please enter some text to enhance.")
        else:
            messages = prompts.build_enhance_prompt(enhance_input, style=selected_style_enh)
            res = execute_llm_request(messages)
            if res:
                st.session_state["enhance_result"] = res

    # Display Result
    if st.session_state.get("enhance_result"):
        res: GenerationResult = st.session_state["enhance_result"]
        parsed = prompts.parse_enhance_or_polish_response(res.text, mode="enhance")

        st.markdown("---")
        st.markdown("### 🎯 C2 Elevated Version")
        st.markdown(f'<div class="c2-card">{parsed.main_text}</div>', unsafe_allow_html=True)

        # Quick Copy Box
        st.code(parsed.main_text, language=None)

        # Lexical Insights
        if parsed.notes:
            st.markdown(
                f'<div class="insight-box"><b>🔍 Lexical & Syntactic Upgrades</b><br>{parsed.notes}</div>',
                unsafe_allow_html=True,
            )

        # Metadata Badge
        st.markdown(
            f'<div class="stats-pill">⚡ {res.latency_seconds:.2f}s</div>'
            f'<div class="stats-pill">🤖 {res.provider.upper()} / {res.model}</div>'
            f'<div class="stats-pill">🎨 {selected_style_enh}</div>',
            unsafe_allow_html=True,
        )


# =========================================================================== #
# TAB 2: C2 SYNONYMS
# =========================================================================== #

with tab_synonyms:
    st.markdown("#### Discover nuanced C1/C2 alternatives with contextual precision")
    st.caption("Synonyms are context-dependent. Learn exact register, subtle connotation, and why words are not interchangeable.")

    col_syn_style, col_syn_chips = st.columns([1, 2])
    with col_syn_style:
        selected_style_syn = st.selectbox(
            "Target Style Context",
            options=AVAILABLE_STYLES,
            index=0,
            key="syn_style_select",
        )

    with col_syn_chips:
        st.markdown("<div style='font-size: 0.8rem; font-weight: 600; opacity: 0.8; margin-bottom: 0.2rem;'>💡 Sample Words:</div>", unsafe_allow_html=True)
        w1, w2, w3, w4 = st.columns(4)
        if w1.button("ideal", key="syn_chip1", use_container_width=True):
            st.session_state["syn_text_input"] = "ideal"
            st.rerun()
        if w2.button("unreachable", key="syn_chip2", use_container_width=True):
            st.session_state["syn_text_input"] = "unreachable"
            st.rerun()
        if w3.button("important", key="syn_chip3", use_container_width=True):
            st.session_state["syn_text_input"] = "important"
            st.rerun()
        if w4.button("ephemeral", key="syn_chip4", use_container_width=True):
            st.session_state["syn_text_input"] = "ephemeral"
            st.rerun()

    syn_word = st.text_input(
        "Enter a word or short phrase:",
        placeholder="E.g., ideal, unreachable, clear, difficult",
        key="syn_text_input",
    )

    col_syn_btn1, col_syn_btn2, _ = st.columns([1.2, 1, 2])
    with col_syn_btn1:
        generate_syn = st.button("📚 Find C2 Synonyms", type="primary", use_container_width=True, key="btn_syn")
    with col_syn_btn2:
        if st.button("🗑️ Clear", use_container_width=True, key="btn_syn_clear"):
            st.session_state["syn_text_input"] = ""
            st.session_state["syn_result"] = None
            st.rerun()

    if generate_syn:
        if not syn_word.strip():
            st.warning("Please enter a word to look up.")
        else:
            messages = prompts.build_synonyms_prompt(syn_word, style=selected_style_syn, n=6)
            res = execute_llm_request(messages)
            if res:
                st.session_state["syn_result"] = res

    # Display Synonyms Result
    if st.session_state.get("syn_result"):
        res: GenerationResult = st.session_state["syn_result"]
        parsed_syn = prompts.parse_synonyms_response(res.text)

        st.markdown("---")
        st.markdown(f"### 📚 C2 Alternatives for *'{syn_word}'*")

        if parsed_syn.items:
            for idx, item in enumerate(parsed_syn.items, 1):
                st.markdown(
                    f"""
                    <div class="syn-card">
                        <div>
                            <span class="syn-term">{idx}. {item.term}</span>
                            <span class="syn-badge">C2 Level</span>
                        </div>
                        <div style="font-size: 0.96rem; margin-top: 0.35rem;">
                            <b>Nuance & Context:</b> {item.nuance}
                        </div>
                        <div class="syn-example">
                            <b>Example:</b> "{item.example}"
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            # Fallback raw markdown display if custom parsing was partial
            st.markdown(res.text)

        # Interchangeability Note Box
        if parsed_syn.note:
            st.markdown(
                f'<div class="note-box"><b>⚠️ Contextual Interchangeability Note</b><br>{parsed_syn.note}</div>',
                unsafe_allow_html=True,
            )

        # Performance badge
        st.markdown(
            f'<div class="stats-pill">⚡ {res.latency_seconds:.2f}s</div>'
            f'<div class="stats-pill">🤖 {res.provider.upper()} / {res.model}</div>',
            unsafe_allow_html=True,
        )


# =========================================================================== #
# TAB 3: POLISH (Preserve Voice)
# =========================================================================== #

with tab_polish:
    st.markdown("#### Polish and refine while strictly preserving your authentic voice")
    st.caption("Eliminates awkward phrasing, sharpens rhythm, and heightens elegance without changing who you sound like.")

    col_pol_style, col_pol_chips = st.columns([1, 2])
    with col_pol_style:
        selected_style_pol = st.selectbox(
            "Style Nuance",
            options=AVAILABLE_STYLES,
            index=0,
            key="polish_style_select",
        )

    with col_pol_chips:
        st.markdown("<div style='font-size: 0.8rem; font-weight: 600; opacity: 0.8; margin-bottom: 0.2rem;'>💡 Sample Drafts:</div>", unsafe_allow_html=True)
        p1, p2 = st.columns(2)
        if p1.button("Boredom Philosophy", key="pol_chip1", use_container_width=True):
            st.session_state["polish_text_area"] = "Even though I disagree with this philosophy, studying it taught me how to embrace boredom."
            st.rerun()
        if p2.button("Project Reflections", key="pol_chip2", use_container_width=True):
            st.session_state["polish_text_area"] = "I realized that making software is not just about writing code, but about knowing why we make it."
            st.rerun()

    polish_input = st.text_area(
        "Enter your draft prose:",
        height=130,
        placeholder="Paste your paragraph or essay excerpt here...",
        key="polish_text_area",
    )

    col_pol_btn1, col_pol_btn2, _ = st.columns([1.2, 1, 2])
    with col_pol_btn1:
        generate_pol = st.button("🖋️ Polish Prose", type="primary", use_container_width=True, key="btn_polish")
    with col_pol_btn2:
        if st.button("🗑️ Clear", use_container_width=True, key="btn_pol_clear"):
            st.session_state["polish_text_area"] = ""
            st.session_state["polish_result"] = None
            st.rerun()

    if generate_pol:
        if not polish_input.strip():
            st.warning("Please enter text to polish.")
        else:
            messages = prompts.build_polish_prompt(polish_input, style=selected_style_pol)
            res = execute_llm_request(messages)
            if res:
                st.session_state["polish_result"] = res

    # Display Polish Result
    if st.session_state.get("polish_result"):
        res: GenerationResult = st.session_state["polish_result"]
        parsed_pol = prompts.parse_enhance_or_polish_response(res.text, mode="polish")

        st.markdown("---")
        st.markdown("### 🖋️ Polished Version (Authentic Voice Preserved)")
        st.markdown(f'<div class="c2-card">{parsed_pol.main_text}</div>', unsafe_allow_html=True)

        st.code(parsed_pol.main_text, language=None)

        if parsed_pol.notes:
            st.markdown(
                f'<div class="insight-box"><b>✨ Refinement Highlights</b><br>{parsed_pol.notes}</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div class="stats-pill">⚡ {res.latency_seconds:.2f}s</div>'
            f'<div class="stats-pill">🤖 {res.provider.upper()} / {res.model}</div>'
            f'<div class="stats-pill">🎨 {selected_style_pol}</div>',
            unsafe_allow_html=True,
        )


# =========================================================================== #
# TAB 4: ARCHITECTURE & GUIDE
# =========================================================================== #

with tab_about:
    st.markdown("### 🏛️ Architecture & Philosophy")
    st.markdown(
        """
        The **C2 English Writing Assistant** is engineered for writers, academics, researchers, and professionals
        who seek to elevate their English expression to the highest tier of nuance, precision, and elegance.

        #### ⚡ Key Architectural Features
        1. **Zero Local GPU Requirement**: Inference is delegated to state-of-the-art LLM APIs (e.g. Groq running Llama-3.3-70B in milliseconds), eliminating large model downloads and GPU requirements.
        2. **Provider Agnostic**: Seamlessly toggle between Groq, OpenAI, OpenRouter, or local/remote OpenAI-compatible endpoints.
        3. **Nuance-Driven System Instructions**: Avoids dictionary-dump obscurity; enforces natural native elegance, collocational accuracy, and stylistic register.

        #### 🎛️ Three Specialized Modes
        - **Enhance Phrase**: Elevates ordinary sentences to C1/C2 mastery while strictly preserving meaning and intent.
        - **C2 Synonyms**: Explores context-dependent alternatives with subtle connotations, register notes, and interchangeability warnings.
        - **Polish**: Refines and polishes prose while meticulously preserving your authentic personal voice.
        """
    )
