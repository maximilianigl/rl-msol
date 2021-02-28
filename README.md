Repository for "Multitask Soft Option Learning" by Maximilian Igl, Andrew Gambardella, Jinke He,
Nantas Nardelli, N. Siddharth, Wendelin Böhmer and Shimon Whiteson.

Requires openai baselines (https://github.com/openai/baselines) and sacred
(https://github.com/IDSIA/sacred).

Please also add `cuda=False` if you're running without GPUs.

Running MSOL on Taxi
```
python main.py -p with architecture.num_options=4 loss.c_kl_b=0.02 loss.c_kl_b_1=0.1
```
For Distral on Taxi

```
python main.py -p with architecture.num_options=1 loss.c_kl_a=0.02 loss.c_kl_b=0. \
loss.entropy_loss_coef0=0.05 loss.entropy_loss_coef1=0.05 loss.entropy_loss_ceof_test=0.05
```



# Required Packages
How I got it to run from scratch on Mac (there's also a `pip_freeze.txt`)

```
# Use python version 3.6 because gym and baselines rely on old version
pyenv install 3.6.12 && pyenv global 3.6.12
pip install sacred gym torch torchvision torchaudio pyyaml


# Install Openai Baselines (see https://github.com/openai/baselines)
brew install cmake openmpi
pip install tensorflow==1.14

git clone https://github.com/openai/baselines.git && cd baselines
pip install -e .
cd ..

git clone git@github.com:maximilianigl/rl-msol.git && cd rl-msol
```
