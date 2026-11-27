# PhishGuard – Phishing SMS and URL Detection (Python)

A local Flask web application for a defensive cybersecurity college project. It analyzes user-pasted SMS text and URLs using **trained machine-learning models plus explainable rules**, and stores recent scans in SQLite.

## Features
- SMS text analysis for urgency, credential/OTP requests, KYC/account threats, and links.
- URL analysis for IP-based hosts, URL shorteners, long URLs, suspicious keywords, punycode, unusual domain structure, and missing HTTPS.
- Extracts URLs from SMS and analyzes them separately.
- Risk score, verdict, and reasons.
- Local SQLite scan history.
- No external URL is opened or fetched by the application.

## Train the models
Datasets go in `data/` (already there): `SMSSpamCollection`, `PhiUSIIL_Phishing_URL_Dataset.csv`, `url_features_extracted1.csv`.
```bash
pip install -r requirements.txt
python train.py          # trains both; or: python train.py sms / python train.py url
```
Outputs: `models/sms_model.joblib`, `models/url_model.joblib`, `reports/metrics.json`. Restart `app.py` afterwards.

- **SMS model:** TF-IDF (word + character n-grams) + Logistic Regression, trained on SMSSpamCollection (duplicates removed).
- **URL model:** 39 lexical features taken only from the URL string + HistGradientBoosting, trained on the URLs of both URL datasets
  (labels unified to 1 = phishing; both datasets store 1 = legitimate). Page-content columns of PhiUSIIL are not used because the app never downloads pages.
- Final score in the app = 70% model probability + 30% rule score.

## Requirements
- Python 3.9 or newer
- pip

## Run on Windows
```bash
py -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
python app.py
```

## Run on Kali Linux / Ubuntu
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

Open http://127.0.0.1:5000 in your browser.

## API example
```bash
curl -X POST http://127.0.0.1:5000/analyze \\
  -H "Content-Type: application/json" \\
  -d '{"type":"sms","content":"Your account is blocked. Verify now at http://example.test"}'
```

Set `type` to `sms` or `url`.

## Important limitations
- This is a rule-based academic prototype, not a validated production anti-phishing product.
- Risk scores are heuristic indicators, not probabilities.
- A "Likely Safe" result is not a guarantee. Attackers can evade simple rules; legitimate URLs can also look suspicious.
- The app analyzes messages pasted by the user. It does not intercept incoming SMS automatically.
- Do not use real victims' private messages or credentials without informed consent.
- Never test by opening suspected phishing links. Use synthetic examples or an isolated lab.
- Dataset bias: in PhiUSIIL almost no legitimate URL has a path (0.3%) while 55% of phishing URLs do, so `train.py` adds synthetic paths/subdomains to legitimate URLs
  and strips paths from some phishing URLs. This reduces, but does not remove, the bias; some real legitimate deep links (e.g. long news-article URLs) can still be flagged.
- SMSSpamCollection is a 2000s-era UK/English dataset; it will be weaker on Indian-style KYC/OTP/UPI smishing than on classic prize spam.
- Reported accuracy is on a random split of these datasets, not on real-world traffic.
