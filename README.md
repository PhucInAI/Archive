# LJMU_Thesis
LJMU thesis work, focus on medical segmentation with ViTs

## Environment setup
```
    conda create -n FFESNet python=3.10 ipython
    conda activate FFESNet

    conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia
    pip install timm
    pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.3/index.html
    pip install matplotlib seaborn scikit-learn scikit-image jupyter
    pip install tensorflow==2.14.0
    
    pip install -e .
```