"""Download Zenodo poker hand histories and extract 1000 Pluribus .phh files."""

import os
import zipfile
import requests
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ZIP_URL = "https://zenodo.org/api/records/13997158/files/poker-hand-histories.zip/content"
ZIP_PATH = os.path.join(os.path.dirname(__file__), "poker-hand-histories.zip")
NUM_HANDS = 1000


def already_have_data():
    if not os.path.isdir(DATA_DIR):
        return False
    phh_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".phh")]
    return len(phh_files) >= NUM_HANDS


def download_zip():
    print(f"Downloading {ZIP_URL} ...")
    resp = requests.get(ZIP_URL, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(ZIP_PATH, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="Downloading"
    ) as bar:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            bar.update(len(chunk))
    print("Download complete.")


def extract_hands():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Extracting Pluribus hands...")
    count = 0
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        for entry in zf.namelist():
            if count >= NUM_HANDS:
                break
            if "pluribus" in entry.lower() and entry.endswith(".phh"):
                data = zf.read(entry)
                out_name = f"hand_{count:04d}.phh"
                with open(os.path.join(DATA_DIR, out_name), "wb") as f:
                    f.write(data)
                count += 1
    print(f"Extracted {count} hands into {DATA_DIR}/")


def cleanup_zip():
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
        print("Deleted zip file.")


def main():
    if already_have_data():
        print(f"data/ already contains >= {NUM_HANDS} .phh files, skipping.")
        return
    download_zip()
    extract_hands()
    cleanup_zip()


if __name__ == "__main__":
    main()
