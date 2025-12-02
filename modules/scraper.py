import time
import requests
from bs4 import BeautifulSoup
import tldextract

SAFE_UAS = {"User-Agent": "Mozilla/5.0 (compatible; dental-agent/1.0)"}

def _allowed(url: str) -> bool:
    bad_hosts = ["pdfdrive", "sci-hub", "libgen"]
    host = tldextract.extract(url).registered_domain
    return all(b not in host for b in bad_hosts)

def scrape_public_pages(urls, max_pages=5, max_chars=5000, sleep_s=1.0):
    texts = []
    for url in urls[:max_pages]:
        if not _allowed(url):
            continue
        try:
            r = requests.get(url, headers=SAFE_UAS, timeout=20)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            for t in soup(["script","style","nav","header","footer","form"]):
                t.decompose()
            text = " ".join(soup.get_text(separator=" ").split())
            if text:
                texts.append(text[:max_chars])
            time.sleep(sleep_s)
        except Exception:
            continue
    return "\n\n".join(texts)[:max_chars]
