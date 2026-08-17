import base64, io, json, sys, urllib.request
from PIL import Image

PATH = sys.argv[1]
URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = json.load(urllib.request.urlopen(
    "http://127.0.0.1:8000/v1/models", timeout=30))["data"][0]["id"]

img = Image.open(PATH).convert("RGB")
print("  file: %s  %dx%d" % (PATH, img.width, img.height))
# keep the long edge sane; the tower tiles internally
if max(img.size) > 1024:
    img.thumbnail((1024, 1024))
    print("  resized to %dx%d" % (img.width, img.height))

buf = io.BytesIO()
img.save(buf, format="PNG")
uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def ask(q, max_tokens=200):
    body = {"model": MODEL, "max_tokens": max_tokens, "temperature": 0.0,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": uri}},
                {"type": "text", "text": q}]}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


for q in ("Describe this image in two or three sentences.",
          "What art style is this, and what is the subject doing?"):
    print("\n  Q: %s\n  A: %s" % (q, ask(q)))
