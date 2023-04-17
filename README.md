# viAct
Repository to store all the code from viAct

# Environment setup
- OS: Ubuntu 22.04.02 LTS
```
    # Create base environment in anaconda
    conda create -n viAct python=3.10 ipython
    conda install -c conda-forge pytorch-gpu=1.13.1 tensorflow-gpu opencv
    conda install -c conda-forge matplotlib seaborn scikit-learn scikit-image tqdm jupyter torchvision

    # Install SAM (Segmnet Anything)
    cd functions/segment-anything
    pip install -e
```

# Current functions
- Check camera reposition
- Check flashlight
- Check lifting region
- Check barrier
- Check crane safety

# Check barrier config