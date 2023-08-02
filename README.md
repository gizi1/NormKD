1.Our code is based on the open-source code of DKD[(Decouple Knowledge Distillation)](https://github.com/megvii-research/mdistiller).\
2.We add new distillers and configs for NormKD and NormKD+DKD.\
3.You can get the pre-trained teacher models, set environment and run the code according to the README.md of [DKD](https://github.com/megvii-research/mdistiller).

For example, you can train the model by NormKD as follow:

```bash

    python3 tools/train.py --cfg configs/cifar100/NormKD/res32x4_res8x4.yaml

```



