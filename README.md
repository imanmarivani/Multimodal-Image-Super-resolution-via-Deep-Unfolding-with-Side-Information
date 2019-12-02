## DMSC-Net 
A tensorflow implementation of DMSC network in [[1]](#ref1)

## LeSITA proximal operator 
LeSITA is a learnable proximal operator for solving l1-l1 minimization problem introduced in [[2]](#ref2). 
```
Two implementations of LeSITA can be found in utils.py
These implementations are first used as activation layers in 
a multimodal feed forward neural network presented in [[1]](#ref1) 
and a deep multimodal CNN for image restoration introduced in[[2]](#ref3). 
```

## References
1. <a name="ref1"></a>I. Marivani, E. Tsiligianni, B. Cornelis and N. Deligiannis, "Multimodal Image Super-resolution via Deep Unfolding with Side Information," 2019 27th European Signal Processing Conference (EUSIPCO), A Coruna, Spain, 2019, pp. 1-5.
2. <a name="ref2"></a>E. Tsiligianni and N. Deligiannis, "Deep Coupled-Representation Learning for Sparse Linear Inverse Problems With Side Information," in IEEE Signal Processing Letters, vol. 26, no. 12, pp. 1768-1772, Dec. 2019.
3. <a name="ref3"></a>I. Marivani, E. Tsiligianni, B. Cornelis and N. Deligiannis, "Learned Multimodal Convolutional Sparse Coding for Guided Image Super-Resolution," 2019 IEEE International Conference on Image Processing (ICIP), Taipei, Taiwan, 2019, pp. 2891-2895.
