Repository for "Multitask Soft Option Learning" by Maximilian Igl, Andrew Gambardella, Jinke He,
Nantas Nardelli, N. Siddharth, Wendelin Böhmer and Shimon Whiteson.


WARNING: There have been some breaking changes (e.g. to gym) so currently the code doesn't run as
is.

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



