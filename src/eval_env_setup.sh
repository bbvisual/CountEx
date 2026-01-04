# for cuda 12.1
# conda create -n countex python=3.10.18
# conda activate countex
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.42.0
pip install deepspeed==0.17.0
pip install accelerate==1.6.0
pip install wandb
pip install datasets
pip install matplotlib
pip install scipy
# gcc-11
conda install -c conda-forge gcc=11 gxx=11
