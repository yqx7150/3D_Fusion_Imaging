# Improved-CNN for Fusion Imaging in Photoacoustic Applications

optical-resolution photoacoustic microscopy (OR-PAM) suffers from a limited depth-of-field (DoF) due to the strong laser beam focusing required for high-resolution imaging. To address this issue, we propose a novel volumetric photoacoustic information fusion method based on a 3D siamese multi-level features convolutional neural network (3DSMFCNN) to achieve large-volume imaging at a low cost.



## System and Fusion Process

- (a) Self-developed Photoacoustic microscopy imaging system  
- (b) Fusion process based on the proposed 3DSMFCNN focused detection  

![Self-developed System](Fig1)



## Focus Detection and Network Structure

- (a) Detailed focus detection and fusion strategy  
- (b) Branch structure of the 3DSMFCNN  

![Focus Detection and Network](Fig2)



## MAP Images

1. MAP images of the multi-focus fiber   
   ![Multi-Focus Fiber Images](Fig3)

2. MAP images of multi-focus vasculature at five noise levels.  
   ![Multi-Focus Vasculature](Fig4)

3. MAP images of 20 μm diameter tungsten wires before and after fusion.   
   ![Tungsten Wires](Fig5)

## Other Related Projects

* Accelerated model-based iterative reconstruction strategy for sparse-view photoacoustic tomography aided by multi-channel autoencoder priors  [<font size=5>**[Paper]**</font>](https://onlinelibrary.wiley.com/doi/10.1002/jbio.202300281)  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/PAT-MDAE)

* Sparse-view reconstruction for photoacoustic tomography combining diffusion model with model-based iteration  [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/pii/S2213597923001118)  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/PAT-Diffusion)

* Unsupervised disentanglement strategy for mitigating artifact in photoacoustic tomography under extremely sparse view  [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/pii/S2213597924000302?via%3Dihub)  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/PAT-ADN)

* Score-based generative model-assisted information compensation for high-quality limited-view reconstruction in photoacoustic tomography  [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/pii/S2213597924000405)  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/Limited-view-PAT-Diffusion)

* Generative model for sparse photoacoustic tomography artifact removal  [<font size=5>**[Paper]**</font>](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12745/1274503/Generative-model-for-sparse-photoacoustic-tomography-artifact-removal/10.1117/12.2683128.short?SSO=1)

* PAT-public-data from NCU  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/PAT-public-data)

* GAID-PAT  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/GAID-PAT)