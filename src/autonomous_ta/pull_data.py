import requests

from bs4 import BeautifulSoup
from pathlib import Path

base_url_dict = {"openstax": "https://openstax.org"}


def get_table_of_contents(base, book):
    toc_url = base + "/books/" + book + "/pages"
    response = requests.get(toc_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    toc_links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith(f"/books/{book}/pages/"):
            slug = href.split("/pages/")[1]
            if slug not in toc_links:
                toc_links.append(slug)
    return toc_links


def download_book(base, book):
    output_dir = Path("data/raw/openstax_statistics")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_url = base_url_dict[base.lower()]
    book = book.lower().replace(" ", "-")
    toc_links = get_table_of_contents(base_url, book)

    for toc in toc_links:
        url = f"{base_url}/books/{book}/pages/{toc}"
        out_file = output_dir / f"{toc}.html"
        print(f"Downloading {url} → {out_file}")
        r = requests.get(url)
        if r.status_code == 200:
            out_file.write_text(r.text, encoding="utf-8")
        else:
            print(f"Failed: {r.status_code}")

    print(f"Downloaded {len(toc_links)} pages")
    