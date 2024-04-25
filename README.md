# MTI DS Minima

## Environment setup
```
# ########################################################################
# Mac OS setup
# ########################################################################
conda create -n ds_minima python=3.11 ipython
conda activate ds_minima

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install pandas opencv-python
pip install matplotlib seaborn scikit-learn scikit-image jupyter
pip install tensorflow-cpu

# ########################################################################
# Linux setup - CPU
# ########################################################################
conda create -n ds_minima python=3.11 ipython
conda activate ds_minima

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
pip install tensorflow-cpu
pip install pandas opencv-python
pip install matplotlib seaborn scikit-learn scikit-image jupyter
```