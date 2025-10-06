import requests
from pathlib import Path
import time

def load_token(token_file="/Users/karan/development/Projects/Ortho/photoroom.txt"):
    """Read API token from a file"""
    with open(token_file, "r") as f:
        return f.read().strip()

def remove_background_photoroom(src, dst="no_bg_photoroom", token_file="/Users/karan/development/Projects/Ortho/photoroom.txt", retries=2, delay=2):
    """Send images to Photoroom API for background removal"""
    src, dst = Path(src), Path(dst)
    dst.mkdir(exist_ok=True)
    token = load_token(token_file)

    files = sorted(list(src.glob("*.jpg")) + list(src.glob("*.jpeg")) + list(src.glob("*.png")))
    print(f"Found {len(files)} files in {src}")
    if not files:
        print("⚠️ No matching files found.")
        return

    api_url = "https://sdk.photoroom.com/v1/segment"  # Background removal endpoint

    headers = {
        "x-api-key": token,
    }

    for i, p in enumerate(files, 1):
        print(f"[{i}/{len(files)}] Uploading {p.name} for background removal...")
        success = False

        for attempt in range(1, retries + 1):
            try:
                with open(p, "rb") as f:
                    files_data = {"image_file": f}
                    response = requests.post(api_url, headers=headers, files=files_data, timeout=60)

                if response.status_code == 200:
                    out_path = dst / (p.stem + ".png")
                    with open(out_path, "wb") as f_out:
                        f_out.write(response.content)
                    print(f"✅ Saved {out_path.name}")
                    success = True
                    break
                else:
                    print(f"⚠️ Attempt {attempt} failed ({response.status_code}): {response.text}")
                    time.sleep(delay)

            except Exception as e:
                print(f"⚠️ Attempt {attempt} failed for {p.name}: {e}")
                time.sleep(delay)

        if not success:
            print(f"❌ Skipped {p.name} after {retries} attempts")

    print(f"\n🎯 All done → saved in: {dst}")

if __name__ == "__main__":
    remove_background_photoroom("img")
