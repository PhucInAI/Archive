# Deenn
Deenn Project deployment

# Installation
## Install Pyenv, step by step:

1. Install dependencies for Python: 
```
sudo apt-get install --no-install-recommends make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev libgirepository1.0-dev
```

2. Install Pyenv:
```
curl https://pyenv.run |bash
```

3. Install enviroment into .bashrc 
```
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
```

4. Run ~/.bashrc to echo:
```
exec "$SHELL" 
```

5. Update Pyenv:
```
pyenv update
```

6. Check the version of installed Pyenv:
```
pyenv --version 
```

## Install python3.8.12 based on pyenv
1. Install python 3.8.12 
```
pyenv install 3.8.12
```
2. Set python version to local project folder:\
Jump into the project folder and set the default version of python:
```
 pyenv local 3.8.12
( If you want to using 3.8.12 for global, using "pyenv global 3.8.12" instead
```

## Create a new environment:
1. Create a new environment by using:
```
python -m venv venv
```
2. Activate the environment:
```
source venv/bin/activate
```

## Install GSStreamer
```
    sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio 
```

## Install some build dependencies and gtk:
```
sudo apt install libgirepository1.0-dev gcc libcairo2-dev pkg-config python3-dev gir1.2-gtk-3.0
```
## Upgrade pip:

```
pip install --upgrade pip
```
## Install Pytorch 1.11.0 with cuda113:
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
## Install torch tensorRT 1.1.0:

```
pip install torch_tensorrt==1.1.0 --find-links https://github.com/pytorch/TensorRT/releases/expanded_assets/v1.1.0
```

## Install nvidia tensorT:

```
pip install nvidia-pyindex
pip install nvidia-tensorrt==8.4.1.5
```


## Install packages in the requirement file:
```
pip install -r requirement.txt 
```
(requirement.txt is laid in the same position as common and main)	

## Fix 'Upsample' object has no attribute 'recompute_scale_factor':
```
In /lib/python3.8/site-packages/torch/nn/modules/upsampling.py in line 153-154

Change:

  return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners,
recompute_scale_factor=self.recompute_scale_factor)

To:

return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners,
# recompute_scale_factor=self.recompute_scale_factor
)

```
