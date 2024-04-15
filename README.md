# Masters
Masters courses

```
    # Install java
    sudo apt install default-jre
    sudo apt install default-jdk

    # Get Java path in system
    sudo update-alternatives --config java

    # Declare JAVA_HOME in /etc/environment
    sudo nano /etc/environment
    JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"
    source /etc/environment

```
```
conda create -n masters python=3.11 ipython
conda activate masters

conda install -c conda-forge pytorch-gpu=2.0.0 torchvision tensorflow-gpu opencv
conda install -c conda-forge matplotlib seaborn scikit-learn scikit-image jupyter 
conda install -c conda-forge fairlearn

pip install raiwidgets
pip install pyspark
```



