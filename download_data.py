"""Download MovieLens 'ml-latest-small' into data/ (GroupLens first, GitHub mirror as fallback)."""
import io
import urllib.request
import zipfile
from pathlib import Path

OFFICIAL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
MIRROR = "https://raw.githubusercontent.com/jasenio/MovieLens/main/{}.csv"
FILES = ["ratings", "movies", "tags"]


def download(dest="data"):
    dest = Path(dest)
    dest.mkdir(exist_ok=True)
    if all((dest / f"{f}.csv").exists() for f in FILES):
        return
    try:
        raw = urllib.request.urlopen(OFFICIAL, timeout=30).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            for f in FILES:
                (dest / f"{f}.csv").write_bytes(z.read(f"ml-latest-small/{f}.csv"))
        print("Downloaded from GroupLens")
    except Exception as e:
        print(f"GroupLens unavailable ({type(e).__name__}); using GitHub mirror")
        for f in FILES:
            urllib.request.urlretrieve(MIRROR.format(f), dest / f"{f}.csv")


if __name__ == "__main__":
    download()
