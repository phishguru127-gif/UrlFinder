"""Lexical URL features. Uses ONLY the raw URL string, so it works on any link
pasted into the app (no page download needed)."""
import math, re, ipaddress
from collections import Counter
from urllib.parse import urlparse

SHORTENERS = {"bit.ly","tinyurl.com","t.co","goo.gl","is.gd","ow.ly","buff.ly","cutt.ly","rb.gy","shorturl.at"}
KEYWORDS = ["login","signin","verify","update","secure","account","bank","confirm","wallet","payment",
            "password","refund","free","bonus","prize","support","webscr","paypal","kyc","otp"]
RISKY_TLDS = {"xyz","top","tk","gq","ml","ga","cf","click","link","work","icu","cyou","buzz","rest","live","site","online"}

def _entropy(s):
    if not s: return 0.0
    c = Counter(s); n = len(s)
    return -sum(v/n*math.log2(v/n) for v in c.values())

def parse(url):
    url = str(url).strip()
    p = urlparse(url if "://" in url else "http://" + url)
    return url, p

FEATURE_NAMES = None

def extract(url):
    url, p = parse(url)
    host = (p.hostname or "").lower()
    path, query = p.path or "", p.query or ""
    try:
        ipaddress.ip_address(host); is_ip = 1
    except ValueError:
        is_ip = 0
    parts = host.split(".") if host else []
    tld = parts[-1] if len(parts) > 1 else ""
    n = max(len(url), 1)
    low = url.lower()
    digits = sum(ch.isdigit() for ch in url)
    letters = sum(ch.isalpha() for ch in url)
    host_digits = sum(ch.isdigit() for ch in host)
    tokens = [t for t in re.split(r"[\W_]+", low) if t]
    f = {
        "url_len": len(url), "host_len": len(host), "path_len": len(path), "query_len": len(query),
        "dots": url.count("."), "host_dots": host.count("."), "hyphens": url.count("-"),
        "host_hyphens": host.count("-"), "underscores": url.count("_"), "slashes": url.count("/"),
        "at": url.count("@"), "qmark": url.count("?"), "eq": url.count("="), "amp": url.count("&"),
        "percent": url.count("%"), "tilde": url.count("~"), "hash": url.count("#"),
        "digits": digits, "digit_ratio": digits/n, "letter_ratio": letters/n,
        "host_digit_ratio": host_digits/max(len(host), 1),
        "special_ratio": sum(not ch.isalnum() for ch in url)/n,
        "is_ip": is_ip, "https": int(p.scheme == "https"), "has_port": int(p.port not in (None, 80, 443)) if _port_ok(p) else 0,
        "subdomains": max(len(parts) - 2, 0), "tld_len": len(tld), "risky_tld": int(tld in RISKY_TLDS),
        "punycode": int("xn--" in host), "shortener": int(host in SHORTENERS),
        "entropy": _entropy(low), "host_entropy": _entropy(host),
        "path_depth": path.count("/"), "double_slash_in_path": int("//" in path),
        "n_tokens": len(tokens), "longest_token": max((len(t) for t in tokens), default=0),
        "kw_hits": sum(k in low for k in KEYWORDS),
        "has_www": int(host.startswith("www.")),
        "file_ext_sus": int(bool(re.search(r"\.(exe|zip|scr|apk|php|html?|js)(\?|$)", path.lower()))),
    }
    return f

def _port_ok(p):
    try:
        p.port; return True
    except ValueError:
        return False

def to_frame(urls):
    import pandas as pd
    return pd.DataFrame([extract(u) for u in urls])
