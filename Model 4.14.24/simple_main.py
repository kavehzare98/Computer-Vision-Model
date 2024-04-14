from ultralytics import YOLO

model = YOLO('last.pt')

results = model(source=0, show=True, conf=0.85, save=True, device='mps')