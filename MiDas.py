import torch
import cv2
from pathlib import Path

# index.html / index2.html と同じ優先順（先に見つかったファイルで depth.png を作る）
IMAGE_CANDIDATES = [Path("input.jpg"), Path("IMG_8098.PNG")]
image_path = next((p for p in IMAGE_CANDIDATES if p.exists()), IMAGE_CANDIDATES[0])

model_type = "DPT_Large"  # 重いのでMacならMiDaS_small推奨

midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
midas.eval()

device = torch.device("cpu")
midas.to(device)

# transform
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# 画像読み込み
img = cv2.imread(str(image_path))
if img is None:
    raise SystemExit(f"画像が読めません: {image_path}（{', '.join(str(p) for p in IMAGE_CANDIDATES)} のいずれかを置いてください）")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(f"depth 推論元: {image_path} ({img_rgb.shape[1]}×{img_rgb.shape[0]})")

# 推論
input_batch = transform(img_rgb).to(device)

with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

# 保存
depth = prediction.cpu().numpy()
depth = (depth - depth.min()) / (depth.max() - depth.min())

cv2.imwrite("depth.png", depth * 255)

print("depth.png saved")
