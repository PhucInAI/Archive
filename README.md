# MTI DS Minima

## Environment setup
```
# ########################################################################
# Mac OS setup
# ########################################################################
conda create -n ds_minima python=3.10 ipython
conda activate ds_minima

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install tensorflow-cpu
pip install pandas opencv-python
pip install matplotlib seaborn scikit-learn scikit-image jupyter

# ########################################################################
# Linux setup - CPU
# ########################################################################
conda create -n ds_minima python=3.10 ipython
conda activate ds_minima

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
pip install tensorflow-cpu
pip install pandas opencv-python
pip install matplotlib seaborn scikit-learn scikit-image jupyter

# ########################################################################
# Linux setup - GPU
# ########################################################################
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow[and-cuda]==2.14.0
pip install pandas opencv-python
pip install matplotlib seaborn scikit-learn scikit-image jupyter
```