from ultralytics import YOLO

# 1. Load a pretrained YOLO26 model (e.g., nano, small, or medium)
model = YOLO("yolo26m.pt") 

# 2. Train the model using the AdamW optimizer
results = model.train(
    data="./datasets/building-1/data.yaml",   # Path to your dataset configuration file
    epochs=200,                   # Number of training epochs
    imgsz=640,                    # Target image size
    batch=8,                     # Batch size (adjust based on your GPU VRAM)
    workers=8,
    optimizer="AdamW",            # Explicitly force the AdamW optimizer
    lr0=0.001,                    # Initial learning rate (recommended lower for AdamW)
    device=0,                      # GPU device ID (e.g., 0) or 'cpu'
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005, 
    patience=50,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    flipud=0.5,
    fliplr=0.5,
    degrees=15.0,
    perspective=0.0005,
    scale=0.5,
    mosaic=1.0,
    copy_paste=0.3
)
