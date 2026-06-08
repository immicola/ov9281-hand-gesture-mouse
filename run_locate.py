import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor
import re

model_path = "/home/micola/models/LocateAnything-3B"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="cuda:0",   # явно на GPU
).eval()

# Берём любую картинку — скачай или укажи свою
img = Image.open("beach.jpg").convert("RGB")

# Ресайз до 1280 по длинной стороне — модель поддерживает до 2.5K, но для теста хватит
MAX_SIZE = 1280
w, h = img.size
if max(w, h) > MAX_SIZE:
    scale = MAX_SIZE / max(w, h)
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    w, h = img.size
    print(f"Resized to {w}x{h}")

messages = [
    {"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "Locate all the instances that matches the following description: person."},
    ]}
]

text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images, videos = processor.process_vision_info(messages)
inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to("cuda")

pixel_values = inputs["pixel_values"].to(torch.bfloat16)

response = model.generate(
    pixel_values=pixel_values,
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    image_grid_hws=inputs.get("image_grid_hws", None),
    tokenizer=tokenizer,
    max_new_tokens=512,
    use_cache=True,
    generation_mode="hybrid",
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
    repetition_penalty=1.1,
    verbose=True,
)

answer = response[0] if isinstance(response, tuple) else response
print("\n=== ОТВЕТ МОДЕЛИ ===")
print(answer)

# Парсим боксы
boxes = []
for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
    x1, y1, x2, y2 = [int(g) for g in m.groups()]
    boxes.append({
        "x1": round(x1 / 1000 * w),
        "y1": round(y1 / 1000 * h),
        "x2": round(x2 / 1000 * w),
        "y2": round(y2 / 1000 * h),
    })

print(f"\n=== НАЙДЕНО ОБЪЕКТОВ: {len(boxes)} ===")
for i, b in enumerate(boxes):
    print(f"  [{i+1}] ({b['x1']}, {b['y1']}) → ({b['x2']}, {b['y2']})")