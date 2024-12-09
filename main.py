from transformers import XCLIPProcessor, XCLIPModel
import torch
from PIL import Image
import cv2

processor = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch32")
model = XCLIPModel.from_pretrained("microsoft/xclip-base-patch32")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

video_path = "IMG_0600.mp4"
cap = cv2.VideoCapture(video_path)

frames = []
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

num_frames = 8
step = max(1, frame_count // num_frames)

idx = 0
extracted = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if idx % step == 0:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
        extracted += 1
        if extracted == num_frames:
            break
    idx += 1
cap.release()

text = ["a dog running", "a person cooking", "a man with a shirt on"]

inputs = processor(
    text=text,
    videos=frames,
    return_tensors="pt",
    padding=True
)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

video_embeds = outputs.video_embeds   # (1, 512)
text_embeds = outputs.text_embeds     # (1, 3, 512)

# Squeeze the batch dimension from text_embeds
text_embeds = text_embeds.squeeze(0)  # (3, 512)

video_embeds = video_embeds / video_embeds.norm(dim=-1, keepdim=True)
text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

similarity = torch.matmul(video_embeds, text_embeds.transpose(0, 1)) # (1,3)
best_match_idx = similarity[0].argmax().item()
print(f"Best match: {text[best_match_idx]} with score {similarity[0, best_match_idx].item()}")