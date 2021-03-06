## DMSC-Net 
A tensorflow implementation of DMSC network in [[1]](#ref1)

## LeSITA proximal operator 
LeSITA is a learnable proximal operator for solving l1-l1 minimization problem introduced in [[2]](#ref2). 
```
Two types of implementation for LeSITA activation can be found in utils.py
```
Unfolding a convolutional multimodal sparse coding using LeSITA into the form of a deep multimodal CNN for image restoration is first introduced in [[3]](#ref3). 
DMSC network is also a multimodal feed forward neural networks (based on sparse coding with side information) for guided image super-resolution presented in [[1]](#ref1).



## References
1. <a name="ref1"></a>I. Marivani, E. Tsiligianni, B. Cornelis and N. Deligiannis, "Multimodal Image Super-resolution via Deep Unfolding with Side Information," 2019 27th European Signal Processing Conference (EUSIPCO), A Coruna, Spain, 2019, pp. 1-5.
2. <a name="ref2"></a>E. Tsiligianni and N. Deligiannis, "Deep Coupled-Representation Learning for Sparse Linear Inverse Problems With Side Information," in IEEE Signal Processing Letters, vol. 26, no. 12, pp. 1768-1772, Dec. 2019.
3. <a name="ref3"></a>I. Marivani, E. Tsiligianni, B. Cornelis and N. Deligiannis, "Learned Multimodal Convolutional Sparse Coding for Guided Image Super-Resolution," 2019 IEEE International Conference on Image Processing (ICIP), Taipei, Taiwan, 2019, pp. 2891-2895.
