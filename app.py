import re
import ipaddress
import sqlite3
from datetime import datetime
from urllib.parse import urlparse
from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
import features as F

app = Flask(__name__)
DB_PATH = "detections.db"

URL_RE = re.compile(r'(?i)\b(?:https?://|www\.)[^\s<>"\']+')
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "ow.ly", "buff.ly", "cutt.ly"}
SUSPICIOUS_WORDS = {
    "login": 2, "verify": 2, "verification": 2, "update": 1, "password": 2,
    "bank": 1, "account": 1, "secure": 1, "confirm": 1, "wallet": 1,
    "payment": 1, "refund": 1, "free": 1, "bonus": 1, "prize": 2
}
SMS_PHRASES = {
    "urgent": 2, "immediately": 2, "account blocked": 3, "account suspended": 3,
    "will be blocked": 3, "verify your account": 3, "click here": 2,
    "claim now": 2, "won a prize": 3, " otp": 1, "share your otp": 5,
    "send otp": 4, "update kyc": 3, "kyc expired": 3, "limited time": 1,
    "payment failed": 1, "delivery failed": 1
}

# ---------- trained models (created by train.py); the app still works without them ----------
BASE = Path(__file__).parent
SMS_MODEL = joblib.load(BASE / "models/sms_model.joblib") if (BASE / "models/sms_model.joblib").exists() else None
_url_bundle = joblib.load(BASE / "models/url_model.joblib") if (BASE / "models/url_model.joblib").exists() else None
URL_MODEL = _url_bundle["model"] if _url_bundle else None
URL_COLS = _url_bundle["columns"] if _url_bundle else None
ML_WEIGHT = 0.7   # final score = 70% trained model + 30% rule-based score


def ml_url_prob(url):
    if URL_MODEL is None:
        return None
    return float(URL_MODEL.predict_proba(F.to_frame([url])[URL_COLS])[0, 1])


def ml_sms_prob(text):
    if SMS_MODEL is None:
        return None
    return float(SMS_MODEL.predict_proba([text])[0, 1])


def blend(rule_score, prob):
    if prob is None:
        return rule_score
    return int(round(ML_WEIGHT * prob * 100 + (1 - ML_WEIGHT) * rule_score))


def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                input_type TEXT NOT NULL,
                content TEXT NOT NULL,
                verdict TEXT NOT NULL,
                score INTEGER NOT NULL,
                reasons TEXT NOT NULL
            )
        """)

def normalize_url(raw):
    raw = raw.strip().rstrip(".,);!?]")
    if raw.lower().startswith("www."):
        raw = "http://" + raw
    return raw

def analyze_url(raw):
    url = normalize_url(raw)
    reasons, score = [], 0
    parsed = urlparse(url if "://" in url else "http://" + url)
    host = (parsed.hostname or "").lower()

    if not host:
        return {"url": url, "verdict": "Suspicious", "score": 50,
                "reasons": ["The URL could not be parsed reliably."]}

    try:
        ipaddress.ip_address(host)
        score += 3
        reasons.append("Uses an IP address instead of a normal domain.")
    except ValueError:
        pass

    if len(url) > 100:
        score += 1
        reasons.append("Unusually long URL.")
    if "@" in parsed.netloc:
        score += 3
        reasons.append("Contains an @ symbol in the address section.")
    if host.startswith("xn--") or ".xn--" in host:
        score += 2
        reasons.append("Contains an internationalized/punycode domain.")
    if host.count("-") >= 2:
        score += 1
        reasons.append("Domain contains several hyphens.")
    if host.count(".") >= 4:
        score += 1
        reasons.append("Contains an unusually deep subdomain structure.")
    if host in SHORTENERS:
        score += 2
        reasons.append("Uses a URL-shortening service that hides the final destination.")
    if parsed.scheme != "https":
        score += 1
        reasons.append("Does not use HTTPS.")
    lower_url = url.lower()
    for word, points in SUSPICIOUS_WORDS.items():
        if word in lower_url:
            score += points
            reasons.append(f"Contains the potentially deceptive keyword: {word}.")
    if len(reasons) == 0:
        reasons.append("No obvious warning signs were found by the basic checks.")

    score = min(score * 12, 100)
    prob = ml_url_prob(url)
    if prob is not None:
        score = blend(score, prob)
        reasons.insert(0, f"Trained URL model: {prob*100:.0f}% probability of phishing.")
    verdict = "Phishing" if score >= 60 else "Suspicious" if score >= 30 else "Likely Safe"
    return {"url": url, "verdict": verdict, "score": score, "reasons": reasons, "ml_probability": prob}

def extract_urls(message):
    found = [normalize_url(x) for x in URL_RE.findall(message or "")]
    # Also catch common bare www links; de-duplicate while preserving order.
    return list(dict.fromkeys(found))

def analyze_sms(message):
    text = (message or "").strip()
    lower = text.lower()
    score, reasons = 0, []

    for phrase, points in SMS_PHRASES.items():
        if phrase in lower:
            score += points
            reasons.append(f"Contains a common social-engineering phrase: '{phrase.strip()}'.")
    if re.search(r'\b(?:otp|one.time.password|pin)\b', lower):
        score += 2
        reasons.append("Mentions an OTP, PIN, or one-time password.")
    if re.search(r'\b(?:http://|https://|www\.)', lower):
        score += 2
        reasons.append("Contains a web link.")
    if re.search(r'\b(?:\d{4,}|₹|rs\.?)\b', lower):
        score += 1
        reasons.append("Contains a monetary or numeric prompt.")
    if text.count("!") >= 2:
        score += 1
        reasons.append("Uses repeated exclamation marks to create urgency.")

    urls = extract_urls(text)
    url_results = [analyze_url(u) for u in urls]
    for result in url_results:
        if result["verdict"] == "Phishing":
            score += 4
            reasons.append("An extracted URL has multiple phishing indicators.")
        elif result["verdict"] == "Suspicious":
            score += 2
            reasons.append("An extracted URL has suspicious characteristics.")

    score = min(score * 10, 100)
    prob = ml_sms_prob(text)
    if prob is not None:
        score = blend(score, prob)
        reasons.insert(0, f"Trained SMS model: {prob*100:.0f}% probability of spam/smishing.")
    verdict = "Phishing" if score >= 60 else "Suspicious" if score >= 30 else "Likely Safe"
    if not reasons:
        reasons.append("No common phishing phrases or obvious links were detected.")
    return {"verdict": verdict, "score": score, "reasons": reasons, "urls": url_results, "ml_probability": prob}

def save_result(kind, content, result):
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO history(created_at,input_type,content,verdict,score,reasons) VALUES(?,?,?,?,?,?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), kind, content[:2000],
             result["verdict"], result["score"], " | ".join(result["reasons"]))
        )

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or request.form
    kind = data.get("type", "sms")
    content = (data.get("content") or "").strip()
    if not content:
        return jsonify({"error": "Please enter a message or URL to analyze."}), 400
    if len(content) > 10000:
        return jsonify({"error": "Input is too long. Please use fewer than 10,000 characters."}), 400

    if kind == "url":
        result = analyze_url(content)
    elif kind == "sms":
        result = analyze_sms(content)
    else:
        return jsonify({"error": "Unknown analysis type."}), 400

    save_result(kind, content, result)
    return jsonify(result)

@app.route("/history")
def history():
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM history ORDER BY id DESC LIMIT 30").fetchall()
    return render_template("history.html", rows=rows)

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "message": "Phishing detection service is running."})

if __name__ == "__main__":
    init_db()
    app.run(host="127.0.0.1", port=5000, debug=True)
