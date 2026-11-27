"""Train the SMS-spam and phishing-URL models.

Run:  python train.py            (both)
      python train.py sms        (SMS only)
      python train.py url        (URL only)

Label convention used everywhere below:  1 = phishing / spam,  0 = legitimate / ham.
(Both URL datasets store it the other way round: label 1 = legitimate, so we flip it.)
"""
import sys, json, time
import numpy as np, pandas as pd, joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report, roc_auc_score)
import features as F

SEED = 42
REPORT = {}


def metrics(y, pred, proba=None):
    m = dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred),
             recall=recall_score(y, pred), f1=f1_score(y, pred),
             confusion_matrix=confusion_matrix(y, pred).tolist())
    if proba is not None:
        m["roc_auc"] = roc_auc_score(y, proba)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}


# ------------------------------------------------------------------ SMS
def train_sms():
    print("\n=== SMS model (SMSSpamCollection) ===")
    df = pd.read_csv("data/SMSSpamCollection", sep="\t", header=None,
                     names=["label", "text"], encoding="latin-1")
    df = df.drop_duplicates("text").reset_index(drop=True)      # exact duplicates leak across splits
    df["y"] = (df.label == "spam").astype(int)
    print("rows after de-dup:", len(df), "| spam share:", round(df.y.mean(), 3))

    Xtr, Xte, ytr, yte = train_test_split(df.text, df.y, test_size=0.2,
                                          stratify=df.y, random_state=SEED)
    vec = lambda: make_union(
        TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(2, 4), min_df=3, sublinear_tf=True))
    candidates = {
        "naive_bayes": make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                                     MultinomialNB(alpha=0.1)),
        "logreg": make_pipeline(vec(), LogisticRegression(C=10, class_weight="balanced", max_iter=2000)),
    }
    results, best, best_f1 = {}, None, -1
    for name, pipe in candidates.items():
        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        results[name] = metrics(yte, (proba >= 0.5).astype(int), proba)
        print(name, results[name])
        if results[name]["f1"] > best_f1:
            best, best_f1, best_name = pipe, results[name]["f1"], name
    print("best:", best_name)
    print(classification_report(yte, best.predict(Xte), target_names=["ham", "spam"], digits=4))
    best.fit(df.text, df.y)                                    # refit on all data for deployment
    joblib.dump(best, "models/sms_model.joblib")
    REPORT["sms"] = dict(best=best_name, test_results=results, n_rows=len(df))


# ------------------------------------------------------------------ URL
def load_urls():
    a = pd.read_csv("data/PhiUSIIL_Phishing_URL_Dataset.csv", usecols=["URL", "label"])
    a["y"] = 1 - a.label                                       # dataset: 1 = legit  -> flip
    a["source"] = "phiusiil"
    b = pd.read_csv("data/url_features_extracted1.csv", usecols=["URL", "ClassLabel"]).dropna()
    b["y"] = 1 - b.ClassLabel.astype(int)                      # dataset: 1 = legit  -> flip
    b["source"] = "url_features"
    df = pd.concat([a[["URL", "y", "source"]], b[["URL", "y", "source"]]], ignore_index=True)
    df["URL"] = df.URL.astype(str).str.strip()
    # a URL that appears with conflicting labels is useless for training -> drop all copies
    conflict = df.groupby("URL").y.nunique()
    df = df[~df.URL.isin(conflict[conflict > 1].index)]
    return df.drop_duplicates("URL").reset_index(drop=True)


WORDS = ("news about blog product item page docs guide help support article story world sports tech home "
         "category store shop learn course review post topic video image gallery events careers contact pricing "
         "download search profile report open data python java kotlin travel food health music games").split()


def synth_suffix(rng):
    w = lambda: rng.choice(WORDS)
    alnum = lambda n: "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")) for _ in range(n))
    t = rng.randrange(12)
    if t == 0: return f"/{w()}/{w()}-{w()}"
    if t == 1: return f"/{rng.randint(2015, 2026)}/{rng.randint(1, 12):02d}/{w()}-{w()}-{w()}.html"
    if t == 2: return f"/watch?v={alnum(11)}"
    if t == 3: return f"/dp/{alnum(10)}"
    if t == 4: return f"/questions/{rng.randint(10**6, 10**8)}/{w()}-{w()}-{w()}"
    if t == 5: return f"/wiki/{w().title()}_{w().title()}"
    if t == 6: return f"/products/{w()}-{w()}?id={rng.randint(1, 99999)}"
    if t == 7: return f"/{w()}/{w()}/index.html"
    if t == 8: return f"/search?q={w()}+{w()}&page={rng.randint(1, 20)}"
    if t == 9: return f"/{w()}/{w()}"
    if t == 10: return f"/{rng.randint(1, 9)}/{w()}/{w()}.html"
    return f"/{w()}/{w()}/{w()}-{rng.randint(1, 999)}"


def augment_legit(df, seed=SEED, p_path=0.6, p_nowww=0.25, p_http=0.08, p_sub=0.15, p_strip_phish=0.4):
    """The datasets contain almost no legitimate URLs with a path (0.3%) while 55% of phishing URLs
    have one, so a model would learn 'has a path => phishing'. To remove that shortcut we
    (a) give legitimate URLs realistic paths / subdomains / schemes and
    (b) strip the path from a share of phishing URLs,
    so that path presence is similarly common in both classes."""
    import random
    from urllib.parse import urlparse
    rng = random.Random(seed)
    subs = ["docs", "blog", "en", "support", "news", "shop", "help", "m", "mail", "learn"]
    urls = df.URL.tolist()
    for i, (u, y) in enumerate(zip(urls, df.y.values)):
        if y == 0:
            u = u.rstrip("/")
            if rng.random() < p_nowww: u = u.replace("://www.", "://", 1)
            if rng.random() < p_sub and u.count("/") == 2: u = u.replace("://www.", "://", 1).replace("://", f"://{rng.choice(subs)}.", 1)
            if rng.random() < p_http and u.startswith("https://"): u = "http://" + u[8:]
            if rng.random() < p_path and u.count("/") == 2: u += synth_suffix(rng)
        else:
            if rng.random() < p_strip_phish:
                p = urlparse(u)
                if p.scheme and p.netloc: u = f"{p.scheme}://{p.netloc}"
        urls[i] = u
    out = df.copy(); out["URL"] = urls
    return out


def url_models():
    return {
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, max_leaf_nodes=63, random_state=SEED),
    }


def train_url():
    print("\n=== URL model (PhiUSIIL + url_features_extracted1) ===")
    t = time.time()
    df = augment_legit(load_urls())
    print("unique URLs:", len(df), "| phishing share:", round(df.y.mean(), 3))
    print(df.groupby("source").y.agg(["count", "mean"]))

    X = F.to_frame(df.URL)
    y = df.y.values
    print("features extracted:", X.shape, f"({time.time()-t:.0f}s)")

    # 1) normal random split (optimistic)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    model = url_models()["hist_gradient_boosting"].fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    random_split = metrics(yte, (proba >= 0.5).astype(int), proba)
    print("random 80/20 split :", random_split)

    # 2) cross-dataset test (more honest): train on PhiUSIIL only, test on URLs that appear ONLY in
    #    url_features_extracted1 (mostly phishing/malware links the model has never seen).
    #    The second dataset's legit half is almost entirely a copy of PhiUSIIL, so it adds mostly new phishing.
    cross = {}
    mtr, mte = (df.source == "phiusiil").values, (df.source == "url_features").values
    m = url_models()["hist_gradient_boosting"].fit(X[mtr], y[mtr])
    p = m.predict_proba(X[mte])[:, 1]
    pred = (p >= 0.5).astype(int)
    cross["train_phiusiil__test_url_features_only"] = dict(
        n=int(mte.sum()), phishing_recall=round(float(pred[y[mte] == 1].mean()), 4),
        legit_specificity=round(float(1 - pred[y[mte] == 0].mean()), 4), n_legit=int((y[mte] == 0).sum()))
    print("train PhiUSIIL -> test on new URLs from url_features:", cross)

    # 3) final model on everything, plus feature importances via the random-split model
    from sklearn.inspection import permutation_importance
    imp = permutation_importance(model, Xte.iloc[:5000], yte[:5000], n_repeats=3,
                                 random_state=SEED, scoring="f1")
    top = sorted(zip(X.columns, imp.importances_mean), key=lambda z: -z[1])[:10]
    print("top features:", [(k, round(v, 4)) for k, v in top])

    final = url_models()["hist_gradient_boosting"].fit(X, y)
    joblib.dump({"model": final, "columns": list(X.columns)}, "models/url_model.joblib")

    probe = ["https://github.com/torvalds/linux", "https://en.wikipedia.org/wiki/Phishing",
             "https://www.amazon.in/dp/B08N5WRWNW", "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
             "https://www.nytimes.com/2024/05/01/world/asia/news-story.html", "https://docs.python.org/3/library/re.html",
             "http://paypal.com.verify-login.xyz/signin", "http://192.168.4.12:8080/bin.sh",
             "https://secure-sbi-kyc-update.top/login.php?id=88"]
    pp = final.predict_proba(F.to_frame(probe)[list(X.columns)])[:, 1]
    print("sanity probe (P(phishing)):")
    for u, v in zip(probe, pp): print(f"  {v:.2f}  {u}")
    REPORT.setdefault("url_probe", {u: round(float(v), 3) for u, v in zip(probe, pp)})
    REPORT["url"] = dict(n_rows=len(df), random_split=random_split, cross_dataset=cross,
                         top_features=[(k, round(float(v), 4)) for k, v in top])


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "sms"): train_sms()
    if which in ("all", "url"): train_url()
    with open("reports/metrics.json", "w") as fh:
        json.dump(REPORT, fh, indent=2)
    print("\nSaved models/ and reports/metrics.json")
