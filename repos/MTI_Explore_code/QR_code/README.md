# QR_code exploration

## QR code datasets: https://github.com/BenSouchet/barcode-datasets
## Code references:
<ul>
    <li> Oriented YOLOv5: https://github.com/BossZard/rotation-yolov5
</ul>

## Installation
```
    # Clone Oriented YOLOv5 into this QR_code folder
    git clone https://github.com/BossZard/rotation-yolov5

    # Install Environment
    conda create -n QR_code python=3.10 ipython
    conda activate QR_code
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
    pip install tensorflow[and-cuda]==2.14.0
    pip install pandas opencv-python Cython
    pip install matplotlib seaborn scikit-learn scikit-image scipy jupyter PyYAML tqdm shapely
    pip install pyzbar
```