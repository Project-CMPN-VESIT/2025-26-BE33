import gradio as gr
import torch
import pandas as pd
import sqlite3
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import PyPDF2
import re
import os
import shutil
import whisper
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageClassification

# =====================================================
# LOAD MODELS
# =====================================================
model_id = "arnabdhar/Swin-V2-base-Food"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageClassification.from_pretrained(model_id)
speech_model = whisper.load_model("tiny")

# =====================================================
# DATA
# =====================================================
food_df = pd.read_csv("food_nutrients.csv")
food_df.columns = food_df.columns.str.strip().str.lower()

# =====================================================
# DATABASE
# =====================================================
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS users(
        email TEXT PRIMARY KEY, password TEXT, name TEXT,
        age INTEGER, gender TEXT, condition TEXT,
        cholesterol REAL, glucose REAL, bp TEXT, pdf_path TEXT
    )""")
    conn.commit()
    conn.close()
init_db()

def hash_password(p):
    return hashlib.sha256(p.encode()).hexdigest()

# =====================================================
# PDF SAVE
# =====================================================
def save_pdf(file, email):
    if file is None: return None
    os.makedirs("medical_reports", exist_ok=True)
    path = f"medical_reports/{email.replace('@','_')}.pdf"
    shutil.copy(file.name, path)
    return path

# =====================================================
# PDF PARSER
# =====================================================
def extract_from_pdf(file):
    if file is None: return "", None, None, "", None, None, ""
    text = ""
    reader = PyPDF2.PdfReader(open(file.name, "rb"))
    for page in reader.pages:
        text += page.extract_text() or ""
    text = text.lower()
    name, age, gender, condition = "", None, None, ""
    m = re.search(r"(patient name|name)[:\-]\s*(.*)", text)
    if m: name = m.group(2).strip().title()
    m = re.search(r"age[:\-]\s*(\d+)", text)
    if m: age = int(m.group(1))
    if "male" in text: gender = "Male"
    elif "female" in text: gender = "Female"
    conditions = []
    for c in ["diabetes","hypertension","cholesterol","thyroid","asthma"]:
        if c in text: conditions.append(c.title())
    condition = ", ".join(conditions)
    cholesterol, glucose, bp = None, None, ""
    m = re.search(r"cholesterol[:\-]?\s*(\d+)", text)
    if m: cholesterol = float(m.group(1))
    m = re.search(r"(blood glucose|glucose)[:\-]?\s*(\d+)", text)
    if m: glucose = float(m.group(2))
    m = re.search(r"blood pressure[:\-]?\s*(\d+/\d+)", text)
    if m: bp = m.group(1)
    return name, age, gender, condition, cholesterol, glucose, bp

# =====================================================
# REGISTER
# =====================================================
def register_user(name, email, password, age, gender, condition, cholesterol, glucose, bp, pdf):
    if not name or not email or not password:
        return gr.update(value=_msg_html("error","Name, Email & Password are required")), gr.update(visible=False)
    pdf_path = save_pdf(pdf, email)
    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("INSERT INTO users VALUES (?,?,?,?,?,?,?,?,?,?)",
            (email, hash_password(password), name, age, gender, condition, cholesterol, glucose, bp, pdf_path))
        conn.commit()
        return gr.update(value=_msg_html("success",f"Welcome, {name}! Account created.")), gr.update(visible=False)
    except:
        return gr.update(value=_msg_html("warn","Email already registered. Please login.")), gr.update(visible=False)

# =====================================================
# LOGIN
# =====================================================
def login_user(email, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT condition, name FROM users WHERE email=? AND password=?",
              (email, hash_password(password)))
    user = c.fetchone()
    conn.close()
    if user:
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            {"condition": user[0], "name": user[1]},
            _welcome_html(user[1], user[0])
        )
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        None,
        _msg_html("error", "Invalid credentials. Please try again.")
    )

# =====================================================
# UTIL
# =====================================================
def v(n, k):
    try: return float(n.get(k, 0))
    except: return 0

def _msg_html(kind, msg):
    colors = {"success": ("#0d2e1a","#3ddc65","#3ddc65"), "error": ("#2e0d0d","#ff5f5f","#ff5f5f"), "warn": ("#2e220d","#f5c518","#f5c518")}
    bg, border, color = colors.get(kind, colors["warn"])
    icons = {"success":"✅","error":"❌","warn":"⚠️"}
    return f'<div style="background:{bg};border:1.5px solid {border};border-radius:12px;padding:14px 20px;color:{color};font-weight:600;font-size:15px;margin:8px 0;">{icons.get(kind,"")} {msg}</div>'

def _welcome_html(name, condition):
    return f'''
<div style="background:linear-gradient(135deg,#0d2e1a,#091a0d);border:1.5px solid #3ddc65;border-radius:16px;padding:22px 24px;margin:12px 0;">
  <div style="font-size:28px;font-weight:900;color:#3ddc65;margin-bottom:4px;">Hey, {name}! 👋</div>
  <div style="color:#7acc7a;font-size:14px;">Condition tracked: <span style="color:#fff;font-weight:600;">{condition or "None"}</span></div>
  <div style="color:#4a8c4a;font-size:13px;margin-top:6px;">Upload a food photo or speak your meal below ↓</div>
</div>'''

# =====================================================
# CHARTS
# =====================================================
BG     = "#060e06"
CARD   = "#0f1a0f"
GREEN  = "#3ddc65"
GREEN2 = "#22a846"
AMBER  = "#f5c518"
RED    = "#ff5f5f"
TEXT   = "#c8e6c8"

def _style(ax, fig, title):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)
    ax.set_title(title, color=GREEN, fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(colors=TEXT, labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor("#1e3a1e")
    ax.grid(axis="y", color="#1e3a1e", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

def plot_macros(n):
    fig, ax = plt.subplots(figsize=(4.5, 3))
    vals  = [v(n,"carbohydrates"), v(n,"proteins"), v(n,"total fat")]
    bars  = ax.bar(["Carbs","Protein","Fat"], vals, color=[GREEN,GREEN2,AMBER], edgecolor="none", width=0.5, zorder=3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2, f"{val:.1f}g",
                ha="center", va="bottom", color=TEXT, fontsize=9, fontweight="bold")
    _style(ax, fig, "⚡ Macronutrients")
    plt.tight_layout(pad=0.8)
    return fig

def plot_risk(n):
    fig, ax = plt.subplots(figsize=(4.5, 3))
    vals   = [v(n,"sugars"), v(n,"sodium"), v(n,"saturated fats")]
    thresh = [25, 1500, 15]
    clrs   = [RED if vals[i] > thresh[i] else AMBER for i in range(3)]
    bars   = ax.bar(["Sugar","Sodium","Sat Fat"], vals, color=clrs, edgecolor="none", width=0.5, zorder=3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2, f"{val:.1f}",
                ha="center", va="bottom", color=TEXT, fontsize=9, fontweight="bold")
    _style(ax, fig, "⚠ Risk Nutrients")
    plt.tight_layout(pad=0.8)
    return fig

def plot_calories(n):
    fig, ax = plt.subplots(figsize=(4.5, 3))
    cal  = v(n,"caloric value")
    pct  = min(cal/2000, 1.0)
    clr  = GREEN if pct < 0.4 else (AMBER if pct < 0.7 else RED)
    ax.pie([pct,1-pct], colors=[clr,"#1a2e1a"], startangle=90,
           wedgeprops=dict(width=0.42, edgecolor=BG, linewidth=2))
    ax.text(0, 0, f"{cal:.0f}\nkcal", ha="center", va="center",
            color=GREEN, fontsize=12, fontweight="bold")
    ax.set_title("🔥 Calories", color=GREEN, fontsize=12, fontweight="bold", pad=10)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    patch = mpatches.Patch(color=clr, label=f"{pct*100:.0f}% of 2000 kcal")
    ax.legend(handles=[patch], loc="lower center", fontsize=8,
              facecolor=CARD, edgecolor="#1e3a1e", labelcolor=TEXT)
    plt.tight_layout(pad=0.8)
    return fig

# =====================================================
# HEALTH CHECK
# =====================================================
def health_check(n, condition):
    sugar    = v(n,"sugars")
    sodium   = v(n,"sodium")
    fat      = v(n,"saturated fats")
    calories = v(n,"caloric value")
    warnings, suggestions = [], []
    cond = condition.lower()
    if "diabetes" in cond and sugar > 25:
        warnings.append(f"Sugar is {sugar}g — high for diabetics (recommended < 25g)")
        suggestions.append("Choose low-sugar foods like fruits or yogurt")
    if ("hypertension" in cond or "bp" in cond) and sodium > 1500:
        warnings.append(f"Sodium is {sodium}mg — high for BP patients")
        suggestions.append("Avoid salty processed foods")
    if "cholesterol" in cond and fat > 15:
        warnings.append(f"Saturated fat is {fat}g — may increase cholesterol")
        suggestions.append("Avoid fried or creamy foods")
    if calories > 800:
        suggestions.append("High calorie meal — eat in moderation")
    is_safe = len(warnings) == 0
    return is_safe, warnings, suggestions

def build_result_html(food_name, is_safe, warnings, suggestions, n):
    sugar  = v(n,"sugars")
    sodium = v(n,"sodium")
    fat    = v(n,"saturated fats")
    carbs  = v(n,"carbohydrates")
    prot   = v(n,"proteins")
    cal    = v(n,"caloric value")

    if is_safe:
        verdict_html = '''
<div class="verdict-safe" id="verdict-box">
  <div class="verdict-icon" id="verdict-icon">✅</div>
  <div class="verdict-word" id="verdict-word">GOOD</div>
  <div class="verdict-sub">Safe for your health condition</div>
</div>'''
        trigger = "safe"
    else:
        verdict_html = '''
<div class="verdict-warn" id="verdict-box">
  <div class="verdict-icon" id="verdict-icon">⚠️</div>
  <div class="verdict-word" id="verdict-word">AVOID</div>
  <div class="verdict-sub">Not ideal for your condition</div>
</div>'''
        trigger = "warn"

    warn_html = ""
    if warnings:
        items = "".join(f'<li>{w}</li>' for w in warnings)
        warn_html = f'<div class="section-card warn-card"><div class="section-title">⚠️ Why to avoid</div><ul class="detail-list">{items}</ul></div>'

    sugg_html = ""
    if suggestions:
        items = "".join(f'<li>{s}</li>' for s in suggestions)
        sugg_html = f'<div class="section-card sugg-card"><div class="section-title">💡 Suggestions</div><ul class="detail-list">{items}</ul></div>'

    macros_html = f"""
<div class="section-card">
  <div class="section-title">📊 Nutrition Snapshot</div>
  <div class="macro-grid">
    <div class="macro-pill green"><span class="m-val">{cal:.0f}</span><span class="m-lbl">kcal</span></div>
    <div class="macro-pill teal"><span class="m-val">{carbs:.1f}g</span><span class="m-lbl">Carbs</span></div>
    <div class="macro-pill blue"><span class="m-val">{prot:.1f}g</span><span class="m-lbl">Protein</span></div>
    <div class="macro-pill amber"><span class="m-val">{fat:.1f}g</span><span class="m-lbl">Sat Fat</span></div>
    <div class="macro-pill red"><span class="m-val">{sugar:.1f}g</span><span class="m-lbl">Sugar</span></div>
    <div class="macro-pill gray"><span class="m-val">{sodium:.0f}mg</span><span class="m-lbl">Sodium</span></div>
  </div>
</div>"""

    return f"""
<div class="result-root" data-trigger="{trigger}">
  <div class="food-name-banner">🍱 {food_name.replace('_',' ').title()}</div>
  {verdict_html}
  {macros_html}
  {warn_html}
  {sugg_html}
</div>
"""

# =====================================================
# ANALYZE FOOD
# =====================================================
def analyze_food(img, user):
    if img is None:
        return _msg_html("warn","Please upload a food image first"), None, None, None, None
    inputs = processor(images=img.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    food = model.config.id2label[out.logits.argmax(-1).item()].lower()
    row  = food_df[food_df["food"]==food]
    if row.empty:
        return _msg_html("warn",f"Food '{food}' not found in database"), None, None, None, None
    n = row.iloc[0].to_dict()
    table = pd.DataFrame(n.items(), columns=["Nutrient","Value"])
    condition = user["condition"] if user else "None"
    is_safe, warnings, suggestions = health_check(n, condition)
    result_html = build_result_html(food, is_safe, warnings, suggestions, n)
    return result_html, table, plot_macros(n), plot_risk(n), plot_calories(n)

# =====================================================
# ANALYZE VOICE
# =====================================================
def speech_to_text(audio):
    if audio is None: return ""
    result = speech_model.transcribe(audio)
    return result["text"].lower()

def extract_food(text):
    for food in food_df["food"].values:
        if food in text: return food
    return None

def analyze_voice(audio, user):
    text = speech_to_text(audio)
    food = extract_food(text)
    if not food:
        return _msg_html("warn",f"Could not detect food from: '{text}'"), None, None, None, None
    row = food_df[food_df["food"]==food]
    n   = row.iloc[0].to_dict()
    table = pd.DataFrame(n.items(), columns=["Nutrient","Value"])
    condition = user["condition"] if user else "None"
    is_safe, warnings, suggestions = health_check(n, condition)
    result_html = build_result_html(food, is_safe, warnings, suggestions, n)
    return result_html, table, plot_macros(n), plot_risk(n), plot_calories(n)

# =====================================================
# LOGOUT
# =====================================================
def logout():
    return gr.update(visible=True), gr.update(visible=False), None, ""

# ######################################################
# CSS
# ######################################################
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&family=JetBrains+Mono:wght@400;700&display=swap');

:root {
  --g:  #3ddc65;
  --g2: #22a846;
  --bg: #050e05;
  --bg2:#0a150a;
  --bg3:#0f1e0f;
  --bdr:rgba(61,220,101,0.18);
  --txt:#d4edd4;
  --sub:#6e9e6e;
  --red:#ff5f5f;
  --amb:#f5c518;
  --r:14px;
}

/* RESET */
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container, .main, footer { background: var(--bg) !important; color: var(--txt) !important; font-family: 'Outfit', sans-serif !important; }
footer { display: none !important; }
.gradio-container { max-width: 480px !important; margin: 0 auto !important; padding: 0 !important; }
.contain { padding: 0 !important; }

/* INPUTS */
input, textarea, select {
  background: var(--bg3) !important; border: 1.5px solid var(--bdr) !important;
  border-radius: 12px !important; color: var(--txt) !important;
  font-family: 'Outfit', sans-serif !important; font-size: 15px !important; padding: 12px 16px !important;
  transition: border-color .2s, box-shadow .2s !important;
}
input:focus, textarea:focus { border-color: var(--g) !important; box-shadow: 0 0 0 3px rgba(61,220,101,.14) !important; outline: none !important; }
label span, .label-wrap span { color: var(--sub) !important; font-size: 11px !important; font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: .8px !important; }

/* BLOCKS */
.block, .gr-group, .gr-box, .gradio-group { background: transparent !important; border: none !important; box-shadow: none !important; }
div.gradio-row { gap: 10px !important; }

/* PRIMARY BUTTON */
button.primary, button[variant=primary] {
  background: linear-gradient(135deg,#3ddc65,#22a846) !important;
  color: #000 !important; font-weight: 800 !important; font-size: 15px !important;
  border: none !important; border-radius: 12px !important; padding: 14px !important;
  cursor: pointer !important; letter-spacing: .3px !important;
  transition: transform .15s, box-shadow .15s !important; width: 100% !important;
  font-family: 'Outfit', sans-serif !important;
}
button.primary:hover, button[variant=primary]:hover {
  transform: translateY(-2px) !important; box-shadow: 0 8px 28px rgba(61,220,101,.4) !important;
}
button.primary:active { transform: scale(.97) !important; }

/* SECONDARY BUTTON */
button.secondary, button[variant=secondary] {
  background: var(--bg3) !important; border: 1.5px solid var(--bdr) !important;
  color: var(--txt) !important; border-radius: 12px !important; padding: 13px !important;
  font-family: 'Outfit', sans-serif !important; font-size: 15px !important;
  font-weight: 600 !important; width: 100% !important; cursor: pointer !important;
  transition: border-color .2s, background .2s !important;
}
button.secondary:hover { border-color: var(--g) !important; background: rgba(61,220,101,.07) !important; }

/* DATAFRAME */
.dataframe { border-radius: 12px !important; overflow: hidden !important; }
.dataframe thead th { background: #0f2e0f !important; color: var(--g) !important; font-size: 11px !important; text-transform: uppercase !important; letter-spacing: .5px !important; font-weight: 700 !important; }
.dataframe tbody td { color: var(--txt) !important; font-size: 13px !important; background: var(--bg2) !important; border-color: rgba(61,220,101,.06) !important; }
.dataframe tbody tr:nth-child(even) td { background: var(--bg3) !important; }

/* ===================================================
   CUSTOM COMPONENTS
=================================================== */

/* SPLASH / HEADER */
.nv-header {
  background: linear-gradient(160deg,#091409 0%,#050e05 100%);
  border-bottom: 1px solid var(--bdr);
  padding: 28px 20px 20px;
  text-align: center;
  position: relative;
  overflow: hidden;
}
.nv-header::before {
  content:''; position:absolute; top:-80px; left:50%; transform:translateX(-50%);
  width:260px; height:160px;
  background:radial-gradient(ellipse,rgba(61,220,101,.22) 0%,transparent 70%);
  pointer-events:none;
}
.nv-logo { font-size:42px; margin-bottom:4px; animation: float 3s ease-in-out infinite; display:block; }
.nv-title {
  font-size: 30px; font-weight: 900; letter-spacing: -1px;
  background: linear-gradient(90deg,#3ddc65,#9effc2,#3ddc65);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text; background-size: 200%; animation: shimmer 3s linear infinite;
}
.nv-sub { color: var(--sub); font-size: 13px; margin-top: 4px; }

/* FORM CARD */
.form-card {
  background: var(--bg2); border: 1px solid var(--bdr);
  border-radius: 20px; padding: 24px 20px; margin: 12px 16px;
}
.form-card-title {
  font-size: 18px; font-weight: 800; color: var(--txt); margin-bottom: 18px;
  display: flex; align-items: center; gap: 8px;
}
.form-divider { border: none; border-top: 1px solid var(--bdr); margin: 16px 0; }

/* SWITCH LINK */
.switch-link {
  text-align: center; padding: 14px; color: var(--sub);
  font-size: 14px;
}
.switch-link span { color: var(--g); font-weight: 700; cursor: pointer; }

/* SCAN PAGE HEADER */
.scan-header {
  padding: 20px 20px 0;
  display: flex; align-items: center; justify-content: space-between;
}
.scan-title { font-size: 22px; font-weight: 900; color: var(--txt); }
.logout-btn-wrap button { width: auto !important; padding: 8px 16px !important; font-size: 13px !important; }

/* UPLOAD ZONE */
.upload-wrap {
  margin: 16px;
  background: var(--bg2);
  border: 2px dashed rgba(61,220,101,0.3);
  border-radius: 20px;
  padding: 24px 16px;
  text-align: center;
  transition: border-color .2s, background .2s;
}
.upload-wrap:hover { border-color: var(--g); background: rgba(61,220,101,.04); }
.upload-icon { font-size: 48px; display: block; margin-bottom: 8px; animation: float 3s ease-in-out infinite; }
.upload-label { font-size: 15px; font-weight: 600; color: var(--txt); margin-bottom: 4px; }
.upload-sub { font-size: 12px; color: var(--sub); }

/* ACTION BUTTONS ROW */
.action-row { display: flex; gap: 10px; padding: 0 16px 16px; }
.action-row > div { flex: 1; }

/* RESULT ROOT */
.result-root { animation: slideUp .5s cubic-bezier(.23,1,.32,1); }
@keyframes slideUp { from{opacity:0;transform:translateY(30px)} to{opacity:1;transform:translateY(0)} }

.food-name-banner {
  background: linear-gradient(135deg,#0d2a0d,#091409);
  border: 1px solid var(--bdr);
  border-radius: 14px; margin: 12px 16px 0;
  padding: 16px 20px;
  font-size: 22px; font-weight: 900; color: var(--g);
  letter-spacing: -.5px;
}

/* VERDICT — GOOD */
.verdict-safe {
  margin: 12px 16px;
  background: linear-gradient(135deg,rgba(61,220,101,.18),rgba(34,168,70,.08));
  border: 2px solid rgba(61,220,101,.55);
  border-radius: 20px; padding: 28px 20px;
  text-align: center;
  animation: popIn .55s cubic-bezier(.175,.885,.32,1.275);
}
.verdict-warn {
  margin: 12px 16px;
  background: linear-gradient(135deg,rgba(255,95,95,.14),rgba(180,30,30,.08));
  border: 2px solid rgba(255,95,95,.5);
  border-radius: 20px; padding: 28px 20px;
  text-align: center;
  animation: popIn .55s cubic-bezier(.175,.885,.32,1.275);
}
.verdict-icon { font-size: 52px; margin-bottom: 4px; animation: bounceIn .6s .2s both; display:block; }

.verdict-safe .verdict-word {
  font-size: 64px; font-weight: 900; letter-spacing: -3px; line-height: 1;
  color: #3ddc65;
  text-shadow: 0 0 40px rgba(61,220,101,.7), 0 0 80px rgba(61,220,101,.3);
  animation: glowPulse 2s infinite, bounceIn .6s .1s both;
}
.verdict-warn .verdict-word {
  font-size: 52px; font-weight: 900; letter-spacing: -2px; line-height: 1;
  color: #ff5f5f;
  text-shadow: 0 0 30px rgba(255,95,95,.6);
  animation: bounceIn .6s .1s both;
}
.verdict-sub { font-size: 14px; color: #7acc7a; margin-top: 8px; font-weight: 500; }
.verdict-warn .verdict-sub { color: #cc8080; }

/* SECTION CARDS */
.section-card {
  background: var(--bg2); border: 1px solid var(--bdr);
  border-radius: 16px; padding: 18px 20px; margin: 10px 16px;
}
.warn-card { border-color: rgba(255,95,95,.3); background: rgba(20,5,5,0.6); }
.sugg-card { border-color: rgba(245,197,24,.25); background: rgba(15,12,3,0.6); }
.section-title { font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: .8px; color: var(--sub); margin-bottom: 12px; }
.detail-list { padding-left: 18px; margin: 0; }
.detail-list li { font-size: 14px; color: var(--txt); line-height: 1.8; }
.warn-card .detail-list li { color: #ffaaaa; }
.sugg-card .detail-list li { color: #ffe899; }

/* MACRO GRID */
.macro-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 8px; }
.macro-pill {
  border-radius: 12px; padding: 12px 8px; text-align: center;
  display: flex; flex-direction: column; gap: 2px;
}
.macro-pill.green  { background: rgba(61,220,101,.12); border:1px solid rgba(61,220,101,.25); }
.macro-pill.teal   { background: rgba(20,180,140,.12); border:1px solid rgba(20,180,140,.25); }
.macro-pill.blue   { background: rgba(50,130,220,.12); border:1px solid rgba(50,130,220,.25); }
.macro-pill.amber  { background: rgba(245,197,24,.10); border:1px solid rgba(245,197,24,.25); }
.macro-pill.red    { background: rgba(255,95,95,.10);  border:1px solid rgba(255,95,95,.25); }
.macro-pill.gray   { background: rgba(120,140,120,.10);border:1px solid rgba(120,140,120,.2); }
.m-val { font-size: 18px; font-weight: 800; color: var(--txt); font-family:'JetBrains Mono',monospace; }
.m-lbl { font-size: 10px; color: var(--sub); text-transform: uppercase; letter-spacing: .5px; font-weight: 700; }

/* CHARTS ROW */
.charts-section { padding: 0 16px 24px; }
.charts-title { font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: .8px; color: var(--sub); margin: 16px 0 10px; }

/* FIREWORKS CANVAS */
#fw-canvas {
  position: fixed; top:0; left:0; width:100%; height:100%;
  pointer-events: none; z-index: 9999; display: none;
}

/* ANIMATIONS */
@keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)} }
@keyframes shimmer { 0%{background-position:0%} 100%{background-position:200%} }
@keyframes glowPulse {
  0%,100%{text-shadow:0 0 30px rgba(61,220,101,.7)}
  50%{text-shadow:0 0 70px rgba(61,220,101,1),0 0 120px rgba(61,220,101,.5)}
}
@keyframes popIn { from{opacity:0;transform:scale(.8)} to{opacity:1;transform:scale(1)} }
@keyframes bounceIn {
  0%{opacity:0;transform:scale(.3)} 50%{transform:scale(1.08)} 70%{transform:scale(.95)} 100%{opacity:1;transform:scale(1)}
}
@keyframes ripple { to{transform:scale(2.5);opacity:0} }

/* SCROLLBAR */
::-webkit-scrollbar{width:4px} ::-webkit-scrollbar-track{background:var(--bg)} ::-webkit-scrollbar-thumb{background:#1e4a1e;border-radius:2px}
"""

# ######################################################
# FIREWORKS + RESULT TRIGGER JS
# ######################################################
INJECT_JS = """
<canvas id="fw-canvas"></canvas>
<script>
(function(){
  const C=document.getElementById('fw-canvas');
  const ctx=C.getContext('2d');
  let particles=[], animId=null;
  function resize(){ C.width=window.innerWidth; C.height=window.innerHeight; }
  window.addEventListener('resize',resize); resize();
  const COLS=['#3ddc65','#9effc2','#f5c518','#ffffff','#22ffaa','#aeffcc','#ffdd66'];
  function burst(x,y,count=90){
    for(let i=0;i<count;i++){
      const angle=(Math.PI*2/count)*i, speed=2+Math.random()*6;
      particles.push({x,y,vx:Math.cos(angle)*speed,vy:Math.sin(angle)*speed,
        alpha:1,color:COLS[Math.floor(Math.random()*COLS.length)],
        size:1.5+Math.random()*3.5,decay:0.010+Math.random()*0.014});
    }
  }
  function animate(){
    ctx.clearRect(0,0,C.width,C.height);
    particles=particles.filter(p=>p.alpha>0.02);
    particles.forEach(p=>{
      p.x+=p.vx; p.y+=p.vy; p.vy+=0.07; p.alpha-=p.decay;
      ctx.globalAlpha=p.alpha; ctx.fillStyle=p.color;
      ctx.beginPath(); ctx.arc(p.x,p.y,p.size,0,Math.PI*2); ctx.fill();
    });
    ctx.globalAlpha=1;
    if(particles.length>0) animId=requestAnimationFrame(animate);
    else{ C.style.display='none'; cancelAnimationFrame(animId); }
  }
  window.launchFireworks=function(){
    C.style.display='block'; particles=[];
    const W=C.width, H=C.height;
    const pts=[[W*.25,H*.25],[W*.75,H*.25],[W*.5,H*.15],[W*.15,H*.55],[W*.85,H*.55],[W*.5,H*.45]];
    pts.forEach(([x,y],i)=>setTimeout(()=>burst(x,y),i*160));
    if(animId) cancelAnimationFrame(animId);
    animId=requestAnimationFrame(animate);
  };
  const observer=new MutationObserver(()=>{
    const root=document.querySelector('.result-root');
    if(!root) return;
    const trigger=root.getAttribute('data-trigger');
    if(trigger==='safe') window.launchFireworks();
  });
  observer.observe(document.body,{childList:true,subtree:true});
})();
</script>
"""

HEADER_HTML = """
<div class="nv-header">
  <span class="nv-logo">🍃</span>
  <div class="nv-title">NutriVision</div>
  <div class="nv-sub">AI-powered food & health intelligence</div>
</div>
"""

# ######################################################
# UI
# ######################################################
with gr.Blocks(theme=gr.themes.Base(), css=CSS, title="NutriVision AI") as app:

    gr.HTML(INJECT_JS)
    gr.HTML(HEADER_HTML)

    user_state = gr.State()

    # ==================================================
    # AUTH PAGE  (Register + Login in one scrollable page)
    # ==================================================
    with gr.Column(visible=True) as auth_page:

        with gr.Tabs():

            # ── LOGIN TAB (first, more common) ────────
            with gr.Tab("🔐  Login"):
                gr.HTML('<div class="form-card">')
                gr.HTML('<div class="form-card-title">👋 Welcome back</div>')

                email_l   = gr.Textbox(label="Email",    placeholder="you@example.com")
                pass_l    = gr.Textbox(label="Password", type="password", placeholder="Your password")
                login_msg = gr.HTML()
                login_btn = gr.Button("Login  →", variant="primary")
                gr.HTML('</div>')

            # ── REGISTER TAB ──────────────────────────
            with gr.Tab("📋  Register"):
                gr.HTML('<div class="form-card">')
                gr.HTML('<div class="form-card-title">🌱 Create Account</div>')

                pdf = gr.File(label="📄 Upload Medical Report (auto-fills below)")

                with gr.Row():
                    name_r = gr.Textbox(label="Full Name", placeholder="Aarav Sharma")
                    age_r  = gr.Number(label="Age")

                with gr.Row():
                    gender_r    = gr.Dropdown(["Male","Female","Other"], label="Gender")
                    condition_r = gr.Textbox(label="Condition", placeholder="Diabetes, Hypertension…")

                with gr.Row():
                    chol_r = gr.Number(label="Cholesterol")
                    gluc_r = gr.Number(label="Glucose")
                    bp_r   = gr.Textbox(label="BP", placeholder="120/80")

                gr.HTML('<hr class="form-divider">')

                with gr.Row():
                    email_r = gr.Textbox(label="Email",    placeholder="you@example.com")
                    pass_r  = gr.Textbox(label="Password", type="password", placeholder="Create password")

                reg_msg = gr.HTML()
                reg_btn = gr.Button("Create Account  →", variant="primary")
                gr.HTML('</div>')

                pdf.change(extract_from_pdf, pdf,
                    [name_r, age_r, gender_r, condition_r, chol_r, gluc_r, bp_r])

    # ==================================================
    # SCAN PAGE  (shown after login)
    # ==================================================
    with gr.Column(visible=False) as scan_page:

        # Welcome banner
        welcome_html = gr.HTML()

        # Upload zone label
        gr.HTML("""
        <div class="upload-wrap">
          <span class="upload-icon">📸</span>
          <div class="upload-label">Upload a food photo</div>
          <div class="upload-sub">or use your camera</div>
        </div>""")

        img = gr.Image(type="pil", label="Food Photo", show_label=False)

        gr.HTML("""
        <div style="margin:0 16px 6px;">
          <div style="font-size:13px;font-weight:700;text-transform:uppercase;
               letter-spacing:.8px;color:#6e9e6e;margin-bottom:8px;">🎙️ Or speak your meal</div>
        </div>""")

        audio = gr.Audio(source="microphone", type="filepath",
                         label="Speak what you're eating", show_label=False)

        with gr.Row(elem_classes=["action-row"]):
            analyze_btn = gr.Button("🔬 Scan Food",  variant="primary")
            voice_btn   = gr.Button("🎤 Voice Scan", variant="secondary")

        logout_btn = gr.Button("Logout", variant="secondary")

        # ── RESULT AREA ────────────────────────────────
        res = gr.HTML()  # Big verdict + macro pills rendered here

        # ── CHARTS ─────────────────────────────────────
        gr.HTML('<div class="charts-section"><div class="charts-title">📈 Detailed Charts</div></div>')
        with gr.Row():
            g1 = gr.Plot(show_label=False)
            g2 = gr.Plot(show_label=False)
        g3 = gr.Plot(show_label=False)

        # Full nutrient table (collapsible feel via accordion)
        with gr.Accordion("📊 Full Nutrient Table", open=False):
            table = gr.Dataframe(wrap=True)

    # ==================================================
    # WIRING
    # ==================================================
    reg_btn.click(
        register_user,
        [name_r, email_r, pass_r, age_r, gender_r, condition_r, chol_r, gluc_r, bp_r, pdf],
        [reg_msg, scan_page]
    )

    login_btn.click(
        login_user,
        [email_l, pass_l],
        [auth_page, scan_page, user_state, welcome_html]
    )

    analyze_btn.click(
        analyze_food,
        [img, user_state],
        [res, table, g1, g2, g3]
    )

    voice_btn.click(
        analyze_voice,
        [audio, user_state],
        [res, table, g1, g2, g3]
    )

    logout_btn.click(
        logout,
        [],
        [auth_page, scan_page, user_state, welcome_html]
    )

app.launch(server_name="0.0.0.0", server_port=7860)
