Please follow the instructions to install the required libraries in a new conda environment. Then download the datasets and all the pre-trained models. Activate the environment in the terminal before executing any commands.

NOTE: Each training experiment with default hyper-parameters requires a Tesla V100 32 GB GPU and runs for 3 days.

For training models, follow the commands below:

StarGAN-v2 on celeba_hq

```bash
python main.py --mode train --num_domains 2 --w_hpf 1 \
               --lambda_reg 1 --lambda_sty 1 --lambda_ds 1 --lambda_cyc 1 \
               --train_img_dir data/celeba_hq/train \
               --val_img_dir data/celeba_hq/val
```

```bash
python main.py --mode train --num_domains 2 --w_hpf 1 \
               --alpha 128 --efficient 1 \
               --lambda_reg 1 --lambda_sty 1 --lambda_ds 1 --lambda_cyc 1 \
               --train_img_dir data/celeba_hq/train --batch_size 8 --num_workers 4 --total_iters 100000 --resume_iter 0 --save_every 5000\
               --val_img_dir data/celeba_hq/val --checkpoint_dir expr/checkpoints/tiny_org_celeba_hq --eval_dir expr/eval/tiny_org_celeba_hq
```