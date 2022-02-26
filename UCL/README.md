# Rethinking the Representational Continuity: Towards Unsupervised Continual Learning
This is the *Pytorch Implementation* for the paper Rethinking the Representational Continuity: Towards Unsupervised Continual Learning

## Abstract
<img align="middle" width="700" src="https://github.com/anonycodes/UCL/blob/master/concept.png">

Continual learning (CL) aims to learn a sequence of tasks without forgetting the previously acquired knowledge. However, recent advances in continual learning are restricted to supervised continual learning (SCL) scenarios. Consequently, they are not scalable to real-world applications where the data distribution is often biased and unannotated. In this work, we focus on *unsupervised continual learning (UCL)*, where we learn the feature representations on an unlabelled sequence of tasks and show that the reliance on annotated data is not necessary for continual learning. We conduct a systematic study analyzing the learned feature representations and show that unsupervised visual representations are surprisingly more robust to catastrophic forgetting, consistently achieve better performance, and generalize better to out-of-distribution tasks than SCL. Furthermore, we find that UCL achieves a smoother loss landscape through qualitative analysis of the learned representations and learns meaningful feature representations.
Additionally, we propose Lifelong Unsupervised Mixup (Lump), a simple yet effective technique that leverages the interpolation between the current task and previous tasks' instances to alleviate catastrophic forgetting for unsupervised representations.

__Contribution of this work__
- We attempt to bridge the gap between continual learning and representation learning and tackle the two important problems of continual learning with unlabelled data and representation learning on a sequence of tasks.
- Systematic quantitative analysis show that UCL achieves better performance over SCL with significantly lower catastrophic forgetting on Sequential CIFAR-10, CIFAR-100 and Tiny-ImageNet. Additionally, we evaluate on out of distribution tasks and few-shot continually learning demonstrating the expressive power of unsupervised representations. 
- We provide visualization of the representations and loss landscapes that UCL learns discriminative, human perceptual patterns and achieves a flatter and smoother loss landscape. Furthermore, we propose Lifelong Unsupervised Mixup (Lump) for UCL, which effectively alleviates catastrophic forgetting and provides better qualitative interpretations. 


## Prerequisites
```
$ pip install -r requirements.txt
```

## Run
* __Split CIFAR-10__ experiment with SimSiam
```
$ python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/simsiam_c10.yaml --ckpt_dir ./checkpoints/cifar10_results/ --hide_progress
```

* __Split CIFAR-100__ experiment with SimSiam

```
$ python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/simsiam_c100.yaml --ckpt_dir ./checkpoints/cifar100_results/ --hide_progress
```

* __Split Tiny-ImageNet__ experiment with SimSiam

```
$ python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/simsiam_tinyimagenet.yaml --ckpt_dir ./checkpoints/tinyimagenet_results/ --hide_progress
```

* __Split CIFAR-10__ experiment with BarlowTwins
```
$ python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/barlow_c10.yaml --ckpt_dir ./checkpoints/cifar10_results/ --hide_progress
```

* __Split CIFAR-100__ experiment with BarlowTwins

```
$ python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/barlowm_c100.yaml --ckpt_dir ./checkpoints/cifar100_results/ --hide_progress
```

* __Split Tiny-ImageNet__ experiment with BarlowTwins

```
$ python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/barlowm_tinyimagenet.yaml --ckpt_dir ./checkpoints/tinyimagenet_results/ --hide_progress
```
