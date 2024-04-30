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
conda create -n masters python=3.10 ipython
conda activate masters

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow[and-cuda]==2.14.0
pip install matplotlib seaborn scikit-learn scikit-image jupyter openpyxl
pip install raiwidgets
pip install pyspark
pip install fairlearn
```



