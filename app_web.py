import streamlit as st
import os
from datetime import datetime
import time
import warnings
import requests
from groq import Groq
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import json
import textwrap

# ============================================================
# 1. CONFIGURATION & SETUP
# ============================================================

warnings.filterwarnings("ignore", category=FutureWarning)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in .env file. Please add it.")
    st.stop()

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Create masterpieces folder
if not os.path.exists("masterpieces"):
    os.makedirs("masterpieces")

# ============================================================
# 2. CORE AI FUNCTIONS - COMPLETE FIX
# ============================================================

def generate_with_retry(prompt, temperature=0.85, max_tokens=1200, max_retries=3):
    """
    Generate text using Groq's LLaMA model with automatic retry.
    Increased max_tokens to 1200 for complete outputs.
    Added a "continue" instruction to prevent truncation.
    """
    for attempt in range(max_retries):
        try:
            # MODIFIED: Added forced continuation instruction
            full_prompt = prompt + "\n\nIMPORTANT: Provide your COMPLETE response. Do NOT cut off or abbreviate. Ensure every concept, every description, and every section is FULLY completed before ending."

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a world-renowned creative director and art critic. You always provide COMPLETE, DETAILED, FULLY FINISHED responses. You never cut off or abbreviate your output. You finish every section completely."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,  # Increased to 1200
                top_p=0.95
            )
            result = response.choices[0].message.content
            
            # Check if response is cut off (ends mid-sentence or mid-word)
            if result and not result[-1] in ['.', '!', '?', '"', "'", '`', '\n']:
                # Try to get continuation
                try:
                    continue_prompt = f"Continue from exactly where you left off. Your last words were: '{result[-200:]}'"
                    continue_response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": "Continue the previous response. Do not repeat. Just continue."},
                            {"role": "user", "content": continue_prompt}
                        ],
                        temperature=temperature,
                        max_tokens=500,
                        top_p=0.95
                    )
                    result += "\n\n" + continue_response.choices[0].message.content
                except:
                    pass  # If continuation fails, return what we have
            
            return result
        except Exception as e:
            error_msg = str(e).lower()
            if "rate" in error_msg or "429" in error_msg or "quota" in error_msg:
                wait_time = (2 ** attempt) * 5
                st.warning(f"⏳ Rate limit. Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise e
    raise Exception("Max retries exceeded. Please try again later.")

def generate_with_multiple_models(prompt):
    """Generate responses from multiple models - FIXED with working models only."""
    results = {}
    
    # ONLY WORKING MODELS
    models = {
        "LLaMA 3.3 (Creative)": "llama-3.3-70b-versatile",
        "LLaMA 3.2 (Fast)": "llama-3.2-3b-preview",
        "LLaMA 3.1 (Balanced)": "llama-3.1-8b-instant"
    }
    
    for model_name, model_id in models.items():
        try:
            full_prompt = prompt + "\n\nProvide a COMPLETE response. Finish everything."
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are a creative assistant. Provide COMPLETE responses. Never cut off."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.85,
                max_tokens=800
            )
            results[model_name] = response.choices[0].message.content
        except Exception as e:
            if "decommission" in str(e).lower():
                results[model_name] = "⚠️ This model is no longer available. Please use LLaMA 3.3."
            else:
                results[model_name] = f"⚠️ Error: {str(e)[:150]}..."
    
    return results

def build_creative_prompt(shape, style, mood, colors, feedback, creativity_level):
    """Build an advanced prompt with COMPLETE output instructions - MODIFIED."""
    
    creativity_words = {
        1: "very conventional and safe",
        3: "somewhat imaginative",
        5: "creative and balanced",
        7: "highly imaginative and bold",
        9: "extremely wild, unconventional, and boundary-pushing"
    }
    
    creativity_desc = creativity_words.get(creativity_level, "creative and balanced")
    
    prompt = f"""
You are a world-renowned creative director and art critic with 30 years of experience.

USER REQUIREMENTS:
- Shape: {shape}
- Art Style: {style}
- Mood/Atmosphere: {mood}
- Color Palette: {colors}
- Creativity Level: {creativity_desc} (level {creativity_level}/10)
- User Feedback to Incorporate: {feedback}

TASK:
Generate 3 COMPLETE, DETAILED creative concepts for an artwork based on these requirements.

For EACH concept, provide ALL of the following (do not skip any):
1. A captivating, poetic title
2. A vivid 3-4 sentence visual description with specific details
3. The emotional impact on the viewer (2-3 sentences)
4. Suggested artistic techniques or mediums (at least 3 techniques)

IMPORTANT FORMAT REQUIREMENTS:
- Each concept must be COMPLETE and fully detailed
- Do NOT abbreviate or cut off any section
- Ensure each concept has ALL four parts
- Write in complete paragraphs

OUTPUT FORMAT:
CONCEPT 1:
Title: [full title]
Visual Description: [3-4 detailed sentences]
Emotional Impact: [2-3 sentences]
Techniques: [list of 3+ techniques]

CONCEPT 2:
Title: [full title]
Visual Description: [3-4 detailed sentences]
Emotional Impact: [2-3 sentences]
Techniques: [list of 3+ techniques]

CONCEPT 3:
Title: [full title]
Visual Description: [3-4 detailed sentences]
Emotional Impact: [2-3 sentences]
Techniques: [list of 3+ techniques]

Make each concept distinct and creative. Provide COMPLETE, FINISHED responses. Do NOT cut off.
"""
    return prompt

def build_critic_prompt(shape, style, mood, colors, ideas):
    """Build a prompt for the Critic agent - MODIFIED for complete output."""
    prompt = f"""
You are a strict but fair art critic and creative director.

USER REQUIREMENTS:
- Shape: {shape}
- Art Style: {style}
- Mood: {mood}
- Color Palette: {colors}

CREATIVE IDEAS TO EVALUATE:
{ideas}

TASK - Provide a COMPLETE, DETAILED evaluation:
1. Evaluate EACH concept individually (be specific about strengths and weaknesses)
2. Identify the STRONGEST concept
3. Explain WHY it's the strongest (give at least 3 specific reasons)
4. Suggest ONE concrete improvement to make it even better

IMPORTANT: Provide a COMPLETE response. Evaluate all concepts fully. Do NOT cut off.

FORMAT:
Evaluation of Concept 1: [detailed evaluation]
Evaluation of Concept 2: [detailed evaluation]
Evaluation of Concept 3: [detailed evaluation]
Strongest Concept: [name]
Reasons: [3+ specific reasons]
Suggested Improvement: [one specific improvement]
"""
    return prompt

def build_refinement_prompt(shape, style, mood, colors, feedback, critic_response):
    """Build a prompt for the Refinement Strategist - MODIFIED."""
    prompt = f"""
You are a master artist and creative director who transforms good ideas into masterpieces.

USER REQUIREMENTS:
- Shape: {shape}
- Art Style: {style}
- Mood: {mood}
- Color Palette: {colors}
- User Feedback: {feedback}

CRITIC'S ANALYSIS:
{critic_response}

TASK:
Take the best concept identified by the critic and transform it into a breathtaking final masterpiece.

Create a BEAUTIFUL, COMPLETE 3-4 SENTENCE DESCRIPTION that:
1. Paints a vivid mental image
2. Incorporates the critic's improvement suggestion
3. Captures the essence of the artwork
4. Is poetic and evocative
5. Feels complete and polished
6. Is a FINISHED, COMPLETE description (not cut off)

Provide ONLY the masterpiece description, nothing else. Make it complete.
"""
    return prompt

# ============================================================
# 3. EXPORT FUNCTIONS
# ============================================================

def export_to_json(data):
    return json.dumps(data, indent=2)

def export_to_markdown(data):
    md = f"""# 🎨 Dream Forge - Masterpiece

## 📋 Creative Brief
| Element | Selection |
|---------|-----------|
| Shape | {data.get('shape', 'N/A')} |
| Art Style | {data.get('style', 'N/A')} |
| Mood | {data.get('mood', 'N/A')} |
| Color Palette | {data.get('colors', 'N/A')} |
| Creativity Level | {data.get('creativity', '5')}/10 |
| User Feedback | {data.get('feedback', 'None')} |

## 💡 Visions (Brainstorming)
{data.get('visions', 'N/A')}

## 🔍 Reflection (Critique)
{data.get('reflection', 'N/A')}

## ✨ Final Masterwork
> {data.get('masterwork', 'N/A')}

---
*Generated by Dream Forge Pro*
"""
    return md

def export_to_text(data):
    text = f"""
==================================================
🌟 DREAM FORGE - MASTERPIECE
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
==================================================

CREATIVE BRIEF:
- Shape: {data.get('shape', 'N/A')}
- Art Style: {data.get('style', 'N/A')}
- Mood: {data.get('mood', 'N/A')}
- Color Palette: {data.get('colors', 'N/A')}
- Creativity Level: {data.get('creativity', '5')}/10
- Feedback: {data.get('feedback', 'None')}

------------------------------------------
💡 VISIONS (COMPLETE)
------------------------------------------
{data.get('visions', 'N/A')}

------------------------------------------
🔍 REFLECTION (COMPLETE)
------------------------------------------
{data.get('reflection', 'N/A')}

------------------------------------------
✨ FINAL MASTERWORK (COMPLETE)
------------------------------------------
{data.get('masterwork', 'N/A')}

==================================================
🎨 Dream Forge Pro · you direct · AI creates
==================================================
"""
    return text

# ============================================================
# 4. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Dream Forge Pro",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Inter:wght@300;400;500;600;700&display=swap');
    
    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        25% { background-position: 50% 0%; }
        50% { background-position: 100% 50%; }
        75% { background-position: 50% 100%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes float {
        0% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-15px) rotate(3deg); }
        100% { transform: translateY(0px) rotate(0deg); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(30px) scale(0.95); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    
    @keyframes rainbowText {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #1a1040, #2d1b69, #1a1040, #0f0c29);
        background-size: 500% 500%;
        animation: gradientMove 20s ease infinite;
    }
    
    .main > div { padding: 1rem 2rem; }
    
    .studio-container {
        max-width: 1200px;
        margin: 0 auto;
        position: relative;
        z-index: 1;
    }
    
    .studio-title {
        font-family: 'Playfair Display', serif;
        font-size: 3.5rem;
        font-weight: 700;
        font-style: italic;
        margin: 0;
        line-height: 1.1;
        background: linear-gradient(90deg, #ff6b6b, #feca57, #ffd700, #4ecdc4, #a855f7, #ff6b6b);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: rainbowText 5s ease infinite;
    }
    
    .studio-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 300;
        color: rgba(255,255,255,0.4);
        margin-top: 0.2rem;
        letter-spacing: 0.05em;
    }
    
    .output-block {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.06);
        transition: all 0.3s ease;
        animation: slideIn 0.5s ease-out;
        max-height: 600px;
        overflow-y: auto;
    }
    
    .output-block:hover { transform: translateY(-2px); }
    .output-block.idea { border-left: 5px solid #ff6b6b; background: rgba(255,107,107,0.05); }
    .output-block.critic { border-left: 5px solid #4ecdc4; background: rgba(78,205,196,0.05); }
    .output-block.final { border-left: 5px solid #ffd700; background: rgba(255,215,0,0.06); }
    .output-block.feedback { border-left: 5px solid #a855f7; background: rgba(168,85,247,0.06); }
    
    .output-block::-webkit-scrollbar {
        width: 6px;
    }
    
    .output-block::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
    }
    
    .output-block::-webkit-scrollbar-thumb {
        background: rgba(255,215,0,0.3);
        border-radius: 10px;
    }
    
    .output-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.6rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: rgba(255,255,255,0.3);
        margin-bottom: 0.6rem;
    }
    
    .output-text {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        line-height: 1.8;
        color: rgba(255,255,255,0.85);
        font-weight: 300;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    
    .output-text.final-text {
        font-family: 'Playfair Display', serif;
        font-size: 1.2rem;
        line-height: 1.9;
        font-weight: 400;
        color: white;
        font-style: italic;
    }
    
    .metric-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-bottom: 1.5rem;
        padding: 1rem 1.5rem;
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    .metric-item {
        flex: 1;
        min-width: 100px;
    }
    
    .metric-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: rgba(255,255,255,0.2);
    }
    
    .metric-value {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        font-weight: 500;
        color: rgba(255,255,255,0.85);
        margin-top: 0.1rem;
    }
    
    .css-1d391kg {
        background: rgba(15,12,41,0.9);
        backdrop-filter: blur(30px);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    .sidebar-section-title {
        font-family: 'Inter', sans-serif;
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: rgba(255,255,255,0.25);
        margin-bottom: 0.5rem;
        display: block;
    }
    
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
        color: white !important;
        padding: 0.6rem 1rem !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #ffd700 !important;
        box-shadow: 0 0 30px rgba(255,215,0,0.05) !important;
    }
    
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
        color: white !important;
        padding: 0.3rem 0.8rem !important;
        font-size: 0.9rem !important;
    }
    
    .stSelectbox > div > div > div {
        color: white !important;
    }
    
    .stTextArea > div > div > textarea {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
        color: white !important;
        padding: 0.6rem 1rem !important;
        font-size: 0.9rem !important;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #a855f7 !important;
        box-shadow: 0 0 30px rgba(168,85,247,0.05) !important;
    }
    
    .stSlider > div {
        padding: 0.2rem 0 !important;
    }
    
    .stSlider label {
        color: rgba(255,255,255,0.4) !important;
        font-size: 0.7rem !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, rgba(255,107,107,0.15), rgba(168,85,247,0.15)) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 30px !important;
        color: rgba(255,255,255,0.7) !important;
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        padding: 0.5rem 1rem !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(255,107,107,0.3), rgba(168,85,247,0.3)) !important;
        border-color: rgba(255,215,0,0.2) !important;
        color: white !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(168,85,247,0.15);
    }
    
    .streamlit-expanderHeader {
        color: rgba(255,255,255,0.6) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.85rem !important;
        transition: all 0.3s ease !important;
    }
    
    .streamlit-expanderHeader:hover {
        color: #ffd700 !important;
        transform: translateX(4px);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 8px !important;
        color: rgba(255,255,255,0.5) !important;
        padding: 0.5rem 1.2rem !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.8rem !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255,215,0,0.1) !important;
        color: #ffd700 !important;
    }
    
    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        color: rgba(255,255,255,0.1);
        letter-spacing: 0.05em;
        border-top: 1px solid rgba(255,255,255,0.03);
        margin-top: 2rem;
    }
    
    .footer span {
        background: linear-gradient(90deg, #ff6b6b, #feca57, #ffd700, #4ecdc4, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .success-message {
        padding: 0.75rem 1.2rem;
        background: rgba(78,205,196,0.12);
        border: 1px solid rgba(78,205,196,0.2);
        border-radius: 10px;
        color: rgba(255,255,255,0.8);
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 5. SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 0.5rem 0;">
        <p style="font-family: 'Playfair Display', serif; font-size: 1.4rem; font-style: italic; 
           background: linear-gradient(90deg, #ff6b6b, #feca57, #ffd700, #4ecdc4, #a855f7);
           -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;">
            Dream Forge
        </p>
        <p style="font-family: 'Inter', sans-serif; font-size: 0.65rem; color: rgba(255,255,255,0.2); margin: 0;">
            Pro Studio
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown('<span class="sidebar-section-title">🎯 Input</span>', unsafe_allow_html=True)
    shape_input = st.text_input("Shape", value="star", placeholder="Enter a shape...", label_visibility="collapsed")
    
    st.markdown('<span class="sidebar-section-title" style="margin-top: 0.5rem;">🎨 Creative Direction</span>', unsafe_allow_html=True)
    art_style = st.selectbox(
        "Art Style",
        ["Surrealist", "Cyberpunk", "Watercolor", "Renaissance", "Anime", "Abstract", "Minimalist", "Steampunk", "Expressionist", "Art Deco"],
        label_visibility="collapsed"
    )
    
    mood = st.selectbox(
        "Mood",
        ["Dreamy", "Mysterious", "Joyful", "Dark", "Epic", "Serene", "Chaotic", "Nostalgic", "Ethereal", "Melancholic"],
        label_visibility="collapsed"
    )
    
    color_palette = st.selectbox(
        "Color Palette",
        ["Vibrant", "Warm", "Cool", "Neon", "Earth", "Monochrome", "Pastel", "Metallic"],
        label_visibility="collapsed"
    )
    
    st.markdown('<span class="sidebar-section-title" style="margin-top: 0.5rem;">⚙️ Advanced</span>', unsafe_allow_html=True)
    temperature = st.slider("🎨 Creativity", 0.1, 1.0, 0.85, 0.05)
    creativity_level = st.slider("🚀 Creativity Boost", 1, 10, 5)
    
    st.markdown('<span class="sidebar-section-title" style="margin-top: 0.5rem;">🔬 Multi-Model</span>', unsafe_allow_html=True)
    compare_models = st.checkbox("Compare 3 AI Models", value=False)
    
    st.markdown('<span class="sidebar-section-title" style="margin-top: 0.5rem;">✏️ Feedback</span>', unsafe_allow_html=True)
    feedback = st.text_area(
        "Your Feedback",
        value="Make it more epic and detailed with cosmic elements.",
        placeholder="e.g., Add more fantasy elements, make it darker...",
        label_visibility="collapsed",
        height=60
    )
    
    st.markdown("---")
    generate = st.button("🔥 Forge Masterpiece", use_container_width=True, type="primary")

# ============================================================
# 6. MAIN CONTENT
# ============================================================

st.markdown("""
<div class="studio-container">
    <div style="padding: 1rem 0 0.5rem 0;">
        <h1 class="studio-title">Dream Forge Pro</h1>
        <p class="studio-subtitle">you direct · AI creates · together you make masterpieces</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Metrics Display
st.markdown(f"""
<div class="metric-container">
    <div class="metric-item">
        <div class="metric-label">Shape</div>
        <div class="metric-value" style="color: #ff6b6b;">{shape_input}</div>
    </div>
    <div class="metric-item">
        <div class="metric-label">Style</div>
        <div class="metric-value" style="color: #feca57;">{art_style}</div>
    </div>
    <div class="metric-item">
        <div class="metric-label">Mood</div>
        <div class="metric-value" style="color: #4ecdc4;">{mood}</div>
    </div>
    <div class="metric-item">
        <div class="metric-label">Colors</div>
        <div class="metric-value" style="color: #a855f7;">{color_palette}</div>
    </div>
    <div class="metric-item">
        <div class="metric-label">Creativity</div>
        <div class="metric-value" style="color: #ffd700;">{creativity_level}/10</div>
    </div>
    <div class="metric-item">
        <div class="metric-label">Feedback</div>
        <div class="metric-value" style="color: rgba(255,255,255,0.5); font-size: 0.8rem; font-style: italic;">"{feedback[:40]}{'...' if len(feedback) > 40 else ''}"</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 7. GENERATION LOGIC
# ============================================================

if generate:
    with st.spinner("🎨 Forging your masterpiece..."):
        try:
            # Build the creative prompt
            creative_prompt = build_creative_prompt(
                shape_input, art_style, mood, color_palette, 
                feedback, creativity_level
            )
            
            # Create tabs
            tabs = st.tabs(["✨ Masterwork", "🔬 Model Comparison", "📊 Export"])
            
            # --- TAB 1: Masterwork ---
            with tabs[0]:
                # Step 1: Ideas
                with st.expander("💡 Visions (Brainstorming)", expanded=True):
                    st.markdown('<div class="output-label">💡 AI is generating 3 COMPLETE creative concepts...</div>', unsafe_allow_html=True)
                    ideas = generate_with_retry(creative_prompt, temperature, max_tokens=1500)
                    st.markdown(f"""
                    <div class="output-block idea">
                        <div class="output-label">💡 Visions</div>
                        <div class="output-text">{ideas.replace(chr(10), '<br>')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption("✅ 3 complete creative concepts generated")
                
                # Step 2: Critic
                with st.expander("🔍 Reflection (Critique)", expanded=True):
                    st.markdown('<div class="output-label">🔍 AI is analyzing and critiquing...</div>', unsafe_allow_html=True)
                    critic_prompt = build_critic_prompt(shape_input, art_style, mood, color_palette, ideas)
                    critic = generate_with_retry(critic_prompt, temperature, max_tokens=1000)
                    st.markdown(f"""
                    <div class="output-block critic">
                        <div class="output-label">🔍 Reflection</div>
                        <div class="output-text">{critic.replace(chr(10), '<br>')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption("✅ Complete critique with best concept identified")
                
                # Step 3: Final Masterpiece
                with st.expander("✨ Masterwork (Final)", expanded=True):
                    st.markdown('<div class="output-label">✨ AI is refining the masterpiece...</div>', unsafe_allow_html=True)
                    final_prompt = build_refinement_prompt(shape_input, art_style, mood, color_palette, feedback, critic)
                    final = generate_with_retry(final_prompt, temperature, max_tokens=600)
                    st.markdown(f"""
                    <div class="output-block final">
                        <div class="output-label">✨ Masterwork</div>
                        <div class="output-text final-text">{final}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption("✅ Complete final masterpiece description")
                
                # Feedback acknowledgment
                st.markdown(f"""
                <div class="output-block feedback">
                    <div class="output-label">🎯 Your Influence</div>
                    <div class="output-text" style="font-style: italic; color: rgba(255,255,255,0.6);">
                        "You asked: <span style="color: #ffd700;">{feedback}</span>"
                    </div>
                    <div style="margin-top: 0.3rem; font-size: 0.7rem; color: rgba(255,255,255,0.2);">
                        ✦ Your feedback shaped the final masterpiece
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # --- TAB 2: Model Comparison ---
            with tabs[1]:
                if compare_models:
                    st.markdown("### 🔬 Multi-Model Comparison")
                    st.caption("Comparing responses from 3 different AI models")
                    
                    comparison_prompt = f"Generate a COMPLETE creative concept for a {shape_input} in {art_style} style with {mood} mood. Use {color_palette} colors. Provide a full, detailed description with Title, Visual Description, Emotional Impact, and Techniques."
                    model_results = generate_with_multiple_models(comparison_prompt)
                    
                    cols = st.columns(len(model_results))
                    for idx, (model_name, result) in enumerate(model_results.items()):
                        with cols[idx]:
                            st.markdown(f"#### {model_name}")
                            if "Error" in result or "decommission" in result.lower():
                                st.warning("⚠️ This model is temporarily unavailable")
                                st.caption("Try LLaMA 3.3 for best results")
                            else:
                                st.text_area("", result, height=400, key=f"model_{idx}")
                else:
                    st.info("ℹ️ Enable 'Compare 3 AI Models' in the sidebar to see different creative approaches.")
                    st.markdown("""
                    <div style="padding: 2rem; background: rgba(255,255,255,0.02); border-radius: 12px; text-align: center; color: rgba(255,255,255,0.3); font-family: 'Inter', sans-serif; font-size: 0.9rem;">
                        <span style="font-size: 2rem; display: block; margin-bottom: 0.5rem;">🔬</span>
                        Multi-model comparison shows how different AI models<br>
                        approach the same creative brief
                    </div>
                    """, unsafe_allow_html=True)
            
            # --- TAB 3: Export ---
            with tabs[2]:
                st.markdown("### 📊 Export Masterpiece")
                st.caption("Download your COMPLETE masterpiece in various formats")
                
                data = {
                    "shape": shape_input,
                    "style": art_style,
                    "mood": mood,
                    "colors": color_palette,
                    "creativity": creativity_level,
                    "feedback": feedback,
                    "visions": ideas,
                    "reflection": critic,
                    "masterwork": final,
                    "timestamp": datetime.now().isoformat()
                }
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.download_button(
                        label="📥 JSON",
                        data=export_to_json(data),
                        file_name=f"masterpiece_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True,
                        key="json_download"
                    )
                
                with col2:
                    st.download_button(
                        label="📥 Markdown",
                        data=export_to_markdown(data),
                        file_name=f"masterpiece_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        use_container_width=True,
                        key="md_download"
                    )
                
                with col3:
                    st.download_button(
                        label="📥 Text",
                        data=export_to_text(data),
                        file_name=f"masterpiece_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True,
                        key="txt_download"
                    )
            
            # --- SAVE ---
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"masterpieces/masterwork_{timestamp}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write("=" * 70 + "\n")
                f.write(f"🌟 MASTERWORK GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"🎯 SHAPE: {shape_input}\n")
                f.write(f"🎨 STYLE: {art_style}\n")
                f.write(f"🌊 MOOD: {mood}\n")
                f.write(f"🎨 COLORS: {color_palette}\n")
                f.write(f"🚀 CREATIVITY: {creativity_level}/10\n")
                f.write(f"✏️ FEEDBACK: {feedback}\n\n")
                f.write("-" * 70 + "\n")
                f.write("💡 VISIONS (COMPLETE):\n")
                f.write("-" * 70 + "\n")
                f.write(ideas + "\n\n")
                f.write("-" * 70 + "\n")
                f.write("🔍 REFLECTION (COMPLETE):\n")
                f.write("-" * 70 + "\n")
                f.write(critic + "\n\n")
                f.write("-" * 70 + "\n")
                f.write("✨ MASTERWORK (COMPLETE):\n")
                f.write("-" * 70 + "\n")
                f.write(final + "\n\n")
                f.write("=" * 70 + "\n")
                f.write("🎨 Dream Forge Pro · you direct · AI creates · together you make masterpieces\n")
            
            st.markdown(f"""
            <div class="success-message">
                💾 Preserved · <strong>{filename}</strong>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("💡 Tip: Wait a few seconds and try again. Free tier has rate limits.")


st.markdown("""
<div class="footer">
    Dream Forge Pro · <span>✦</span> · you direct · AI creates · together you make masterpieces
</div>
""", unsafe_allow_html=True)