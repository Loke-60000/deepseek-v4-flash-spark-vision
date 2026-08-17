import base64, io, json, urllib.request
from PIL import Image, ImageDraw

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = json.load(urllib.request.urlopen(
    "http://127.0.0.1:8000/v1/models", timeout=30))["data"][0]["id"]
print("  model:", MODEL)


def ask(messages, max_tokens=120):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0.0}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


def data_uri(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# 1. text still works through the vision wrapper
print("\n  [text] ", ask([{"role": "user", "content": "Reply with exactly: text ok"}], 16))

# 2. an unambiguous synthetic image: red circle, blue square, white bg
img = Image.new("RGB", (448, 448), "white")
d = ImageDraw.Draw(img)
d.ellipse([40, 40, 200, 200], fill="red")
d.rectangle([250, 250, 400, 400], fill="blue")
print("  [shapes]", ask([{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": data_uri(img)}},
    {"type": "text", "text": "What shapes and colors do you see? Answer in one short sentence."},
]}]))

# 3. legible text, to probe OCR
img2 = Image.new("RGB", (448, 200), "white")
d2 = ImageDraw.Draw(img2)
d2.text((40, 80), "GADGET-LAB 42", fill="black")
print("  [ocr]   ", ask([{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": data_uri(img2)}},
    {"type": "text", "text": "What text is written in this image? Reply with only the text."},
]}], 32))
