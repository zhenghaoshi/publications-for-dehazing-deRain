# publications-for-dehazing-deRain
publications for dehazing/deRain
# Image Quality Metrics
* PSNR (Peak Signal-to-Noise Ratio) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4550695)] [[matlab code](https://www.mathworks.com/help/images/ref/psnr.html)] [[python code](https://github.com/aizvorski/video-quality)]
* SSIM (Structural Similarity) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1284395)] [[matlab code](http://www.cns.nyu.edu/~lcv/ssim/ssim_index.m)] [[python code](https://github.com/aizvorski/video-quality/blob/master/ssim.py)]
* VIF (Visual Quality) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1576816)] [[code](https://sse.tongji.edu.cn/linzhang/IQA/Evalution_VIF/eva-VIF.htm)]
* FSIM (Feature Similarity) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5705575)] [[code](https://sse.tongji.edu.cn/linzhang/IQA/FSIM/FSIM.htm)]
* NIQE (Naturalness Image Quality Evaluator) [[paper](http://live.ece.utexas.edu/research/Quality/niqe_spl.pdf)][[matlab code](http://live.ece.utexas.edu/research/Quality/index_algorithms.htm)][[python code](https://github.com/aizvorski/video-quality/blob/master/niqe.py)]

# 1 Dehazing Research
## 1.1 Datasets
* KITTI [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4550695&tag=1)][[data](http://www.cvlibs.net/datasets/kitti/eval_depth_all.php)]
* RESIDE [[paper](https://arxiv.org/pdf/1712.04143.pdf)][[dataset](https://sites.google.com/view/reside-dehaze-datasets)]
* SceneNet [paper][[dataset](https://robotvault.bitbucket.io/scenenet-rgbd.html)]
* I-HAZE [[paper](https://data.vision.ee.ethz.ch/cvl/ntire18//o-haze/O-HAZE.pdf)][[dataset](https://data.vision.ee.ethz.ch/cvl/ntire18//i-haze/)]
* O-HAZE [[paper](https://data.vision.ee.ethz.ch/cvl/ntire18//o-haze/O-HAZE.pdf)][[dataset](https://data.vision.ee.ethz.ch/cvl/ntire18//o-haze/)]
* D-HAZY [[paper](https://www.meo.etc.upt.ro/AncutiProjectPages/D_Hazzy_ICIP2016/D_HAZY_ICIP2016.pdf)][[dataset](https://www.researchgate.net/publication/307516141_D-HAZY_A_dataset_to_evaluate_quantitatively_dehazing_algorithms)]
* Middlebury [[paper](https://skylight.middlebury.edu/~schar/papers/datasets-gcpr2014.pdf)][[dataset](http://vision.middlebury.edu/stereo/data/scenes2014/)]
* NYU Depth Dataset V2 [[paper](https://cs.nyu.edu/~silberman/papers/indoor_seg_support.pdf)][[dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)]
## 1.2 Papers
### 2020
* JinshanPan,Physics-Based Generative Adversarial Models for Image Restoration and Beyond(PAMI 2020)[[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8968618)][code][web]
* Zhenghao Shi,Yaning Feng , Minghua Zhao, Erhu Zhang, Lifeng He, Normalized Gamma Transformation Based Contrast Limited Adaptive Histogram Equalization with Color Correction for Sand-Dust Image Enhancement. IET Image Processing. 14(4):747 -756, 2020 [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9037534)][[code](https://github.com/zhenghaoshi/sand-dust-image-enhancing-with-CLEHE-LAB)]
* Sourya et al, Fast Deep Multi-patch Hierarchical Network for Nonhomogeneous Image Dehazing (CVPR)[[paper](https://arxiv.org/pdf/2005.05999.pdf)][code]
* Dong et al, Multi-Scale Boosted Dehazing Network with Dense Feature Fusion. (CVPR) [[paper](https://arxiv.org/pdf/2004.13388.pdf)][[code](https://github.com/BookerDeWitt/MSBDN-DFF)]
* Li et al, Learning to Dehaze From Realistic Scene with A Fast Physics Based Dehazing Network. [[paper](https://arxiv.org/pdf/2004.08554.pdf)][[code](https://github.com/liruoteng/3DRealisticSceneDehaze)]
* Shao et al, Domain Adaptation for Image Dehazing. (CVPR) [[paper](https://arxiv.org/pdf/2005.04668.pdf)][code][web]
* Wu et al, Accurate Transmission Estimation for Removing Haze and Noise from a Single Image. (TIP) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8891906)][code]
* Ren et al, Single Image Dehazing via Multi-Scale Convolutional Neural Networks with Holistic Edges. (IJCV) [[paper](https://idp.springer.com/authorize?response_type=cookie&client_id=springerlink&redirect_uri=http://link.springer.com/article/10.1007/s11263-019-01235-8)][code]
* Dong et al, FD-GAN: Generative Adversarial Networks with Fusion-discriminator for Single Image Dehazing. [[paper](https://arxiv.org/pdf/2001.06968.pdf)][[code](https://github.com/WeilanAnnn/FD-GAN)]
* Qin et al, FFA-Net: Feature Fusion Attention Network for Single Image Dehazing. (AAAI) [[paper](https://arxiv.org/pdf/1911.07559.pdf)][[code](https://github.com/zhilin007/FFA-Net)]
### 2019
* Wu et al, Learning Interleaved Cascade of Shrinkage Fields for Joint Image Dehazing and Denoising. (TIP) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8852852)][code]
* Li et al, Semi-Supervised Image Dehazing. (TIP) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8902220)][code]
* Li et al, Benchmarking Single Image Dehazing and Beyond. (TIP) [[paper](https://arxiv.org/abs/1712.04143)][code][[web](https://sites.google.com/site/boyilics/website-builder/reside)]
* Pei et al, Classification-driven Single Image Dehazing. [[paper](https://arxiv.org/abs/1911.09389)][code]
* Liu et al, GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing. (ICCV) [[paper](https://arxiv.org/pdf/1908.03245.pdf)][[code](https://github.com/proteus1991/GridDehazeNet)]
* Li et al, Joint haze image synthesis and dehazing with mmd-vae losses. [[paper](https://arxiv.org/pdf/1905.05947.pdf)][code]
* Peter et al, Feature Forwarding for Efficient Single Image Dehazing. [[paper](https://arxiv.org/pdf/1904.09059.pdf)][code]
* Shu et al, Variational Regularized Transmission Refinement for Image Dehazing. [[paper](https://arxiv.org/pdf/1902.07069.pdf)][code]
* Liu et al, End-to-End Single Image Fog Removal using Enhanced Cycle Consistent Adversarial Networks. [[paper](https://arxiv.org/pdf/1902.01374.pdf)][code]
* Chen et al, Gated Context Aggregation Network for Image Dehazing and Deraining. (WACV) [[paper](https://arxiv.org/pdf/1811.08747.pdf)][[code](https://github.com/cddlyf/GCANet)]
* Ren et al, Deep Video Dehazing with Semantic Segmentation. (TIP) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8492451)][code]
### 2018
* Ren et al, Gated Fusion Network for Single Image Dehazing. (CVPR) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8578441)][[code](https://github.com/rwenqi/GFN-dehazing)]
* Zhang et al, FEED-Net: Fully End-To-End Dehazing. (ICME) [paper][code]
* Zhang et al, Densely Connected Pyramid Dehazing Network. (CVPR) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8578435)][[code](https://github.com/hezhangsprinter/DCPDN)]
* Yang et al, Towards Perceptual Image Dehazing by Physics-based Disentanglement and Adversarial Training. (AAAI) [[paper](https://dblp.uni-trier.de/rec/html/conf/aaai/YangXL18)][code]
* Deniz et al, Cycle-Dehaze: Enhanced CycleGAN for Single Image Dehazing. (CVPRW) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8575279)][code]
### Before 2018
* Ren et al, An All-in-One Network for Dehazing and Beyond. (ICCV) [[paper](https://arxiv.org/pdf/1707.06543.pdf)][[code](https://github.com/MayankSingal/PyTorch-Image-Dehazing)]
* Zhu et al, A fast single image haze removal algorithm using color attenuation prior. (TIP) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7128396)][code]
* Cai et al, DehazeNet: An end-to-end system for single image haze removal. (TIP) [[paper](http://caibolun.github.io/papers/DehazeNet.pdf)][[code](https://github.com/caibolun/DehazeNet)]
* Ren et al, Single Image Dehazing via Multi-Scale Convolutional Neural Networks. (ECCV) [paper][code][web]
* He et al, Single Image Haze Removal Using Dark Channel Prior. (CVPR) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5567108)][[web and code](http://kaiminghe.com/cvpr09/index.html)]

# 2 DeRain Research
## 2.1 Single Image Deraining
### 2.1.1 Datasets
### 2.1.1.1 Synthetic Datasets
* Rain12 [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7780668)] [[data](http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)] (2016 CVPR)
* Rain100L_old_version [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yang_Deep_Joint_Rain_CVPR_2017_paper.pdf)][[dataset](http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)](2017 CVPR)
* Rain100L_new_version [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yang_Deep_Joint_Rain_CVPR_2017_paper.pdf)][[data](https://github.com/hezhangsprinter/ID-CGAN)]
* Rain100H_old_version [paper][dataset](2017 CVPR)
* Rain100H_new_version [paper][dataset]
* Rain800 [paper][dataset] (2017 Arxiv)
* Rain1200 [[paper](https://arxiv.org/pdf/1802.07412.pdf)][[dataset](https://github.com/hezhangsprinter/DID-MDN)] (2018 CVPR)
* Rain1400 [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Fu_Removing_Rain_From_CVPR_2017_paper.pdf)][[data](https://xueyangfu.github.io/projects/cvpr2017.html)] (2017 CVPR)
* Heavy Rain Dataset [[paper](http://export.arxiv.org/pdf/1904.05050)][dataset] (2019 CVPR)
### 2.1.1.2 Real-world Datasets
* Practical_by_Yang [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yang_Deep_Joint_Rain_CVPR_2017_paper.pdf)][[data](http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)] (2017 CVPR)
* Practica_by_Zhang [[paper](https://arxiv.org/pdf/1701.05957.pdf)][[data](https://github.com/hezhangsprinter/ID-CGAN)] (2017 Arxiv)
* Real-world Paired Rain Dataset [[paper](https://arxiv.org/pdf/1904.01538.pdf)][[data](https://stevewongv.github.io/derain-project.html)] (2019 CVPR)
## 2.1.2 Papers
### 2020
* Du, Yingjun etc. Conditional Variational Image Deraining. (2020 TIP) [[paper](https://arxiv.org/pdf/2004.11373.pdf)][[code](https://github.com/Yingjun-Du/VID)]
* Jiang Kui et al. Multi-Scale Progressive Fusion Network for Single Image Deraining. (2020 CVPR) [[paper](https://arxiv.org/pdf/2003.10985.pdf)][[code](https://github.com/kuihua/MSPFN)]
* Cong Wang et al. Physical Model Guided Deep Image Deraining. (2020 ICME) [[paper](https://arxiv.org/pdf/2003.13242.pdf)][[code](https://supercong94.wixsite.com/supercong94)]
* Yang, Youzhao et al. RDDAN: A Residual Dense Dilated Aggregated Network for Single Image Deraining. (2020 ICME) [paper][[code](https://github.com/nnUyi/RDDAN)][[web](https://github.com/nnUyi)]
* Ran, Wu; Yang, Youzhao et al. Single Image Rain Removal Boosting via Directional Gradient. (2020 ICME) [paper][[code](https://github.com/nnUyi/DiG-CoM)][[web](https://github.com/nnUyi)]
* Xu, Jun et al. Variational Image Deraining. (2020 WACV) [[paper](http://openaccess.thecvf.com/content_WACV_2020/papers/Du_Variational_Image_Deraining_WACV_2020_paper.pdf)][code]
* Rajeev Yasarla et al. Confidence Measure Guided Single Image De-Raining. (2020 TIP) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9007569)][code]
### 2019
* Yang, Wenhan et al. Single Image Deraining: From Model-Based to Data-Driven and Beyond. (2019 TPAMI) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9096521)][code]
* Yang, Wenhan et al. Scale-Free Single Image Deraining Via VisibilityEnhanced Recurrent Wavelet Learning. (2019 TIP) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8610325)][code]
* Wei, Yanyan et al. A Coarse-to-Fine Multi-stream Hybrid Deraining Network for Single Image Deraining. (2019 ICDM) [[paper](https://arxiv.org/ftp/arxiv/papers/1908/1908.10521.pdf)][code]
* Wang, Hong et al. A Survey on Rain Removal from Video and Single Image. (2019 Arxiv) [[paper](https://arxiv.org/pdf/1909.08326.pdf)][[code](https://github.com/hongwang01/Video-and-Single-Image-Deraining)]
* Wang, Guoqing et al. ERL-Net: Entangled Representation Learning for Single Image De-Raining. (2019 ICCV) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_ERL-Net_Entangled_Representation_Learning_for_Single_Image_De-Raining_ICCV_2019_paper.pdf)][[code](https://github.com/RobinCSIRO/ERL-Net-for-Single-Image-Deraining)]
* Yang, Youzhao et al. Single Image Deraining via Recurrent Hierarchy and Enhancement Network. (2019 ACM'MM) [[paper](https://dl.acm.org/doi/10.1145/3343031.3351149#URLTOKEN#)][[code](https://github.com/nnUyi/ReHEN)]
* Wang, Zheng et al. DTDN: Dual-task De-raining Network. (2019 ACM'MM) [[paper](https://dl.acm.org/doi/pdf/10.1145/3343031.3350945)][code]
* Yu, Weijiang et al. Gradual Network for Single Image De-raining. (2019 ACM'MM) [[paper](https://dl.acm.org/doi/pdf/10.1145/3343031.3350883)][code]
* Wang, Yinglong et al. An Effective Two-Branch Model-Based Deep Network for Single Image Deraining. (2019 Arxiv) [[paper](https://arxiv.org/pdf/1905.05404.pdf)][code]
* Yang, Youzhao el al. Single Image Deraining using a Recurrent Multi-scale Aggregation and Enhancement Network. (2019 ICME) [[paper](https://github.com/nnUyi/ReMAEN/tree/master/paper)][[code](https://github.com/nnUyi/ReMAEN)]
* Liu, Xing et al. Dual Residual Networks Leveraging the Potential of Paired Operations for Image Restoration. (2019 CVPR) [[paper](https://arxiv.org/pdf/1903.08817v1.pdf)][[code](https://github.com/liu-vis/DualResidualNetworks)]
* Li, Ruoteng et al. Heavy Rain Image Restoration: Integrating Physics Model and Conditional Adversarial Learning. (2019 CVPR) [[paper](http://export.arxiv.org/pdf/1904.05050)][[code](https://github.com/liruoteng/HeavyRainRemoval)]
* Wang, Tianyu et al. Spatial Attentive Single-Image Deraining with a High Quality Real Rain Dataset. (2019 CVPR) [[paper](https://arxiv.org/pdf/1904.01538.pdf)][[code](https://github.com/stevewongv/SPANet)][dataset]
* Li, Siyuan et al. Single Image Deraining: A Comprehensive Benchmark Analysis. (2019 CVPR)  [[paper](https://arxiv.org/pdf/1903.08558.pdf)][[code](https://github.com/lsy17096535/Single-Image-Deraining)][[dataset](https://github.com/lsy17096535/Single-Image-Deraining)]
* Hu, Xiaowei et al. Depth-attentional Features for Single-image Rain Removal. (2019 CVPR) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Hu_Depth-Attentional_Features_for_Single-Image_Rain_Removal_CVPR_2019_paper.pdf)][[code](https://github.com/xw-hu/DAF-Net)]
* Wei, Wei et al. Semi-supervised Transfer Learning for Image Rain Removal. (2019 CVPR)  [[paper](https://arxiv.org/pdf/1807.11078.pdf)][[code](https://github.com/wwzjer/Semi-supervised-IRR)]
* Ren, Dongwei et al. Progressive Image Deraining Networks: A Better and Simpler Baseline. (2019 CVPR) [[paper](https://arxiv.org/pdf/1901.09221.pdf)][[code](https://github.com/csdwren/PReNet)]
* Rajeev Yasarla et al. Uncertainty Guided Multi-Scale Residual Learning-using a Cycle Spinning CNN for Single Image De-Raining. (2019 CVPR)  [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yasarla_Uncertainty_Guided_Multi-Scale_Residual_Learning-Using_a_Cycle_Spinning_CNN_for_CVPR_2019_paper.pdf)][[code](https://github.com/rajeevyasarla/UMRL--using-Cycle-Spinning)]
* Zhu, Hongyuan et al. RR-GAN: Single Image Rain Removal Without Paired Information. (2019 AAAI) [[paper](http://vijaychan.github.io/Publications/2019_derain.pdf)][code]
* Fu, Xueyang et al. Lightweight Pyramid Networks for Image Deraining. (2019 TNNLS) [[paper](https://arxiv.org/pdf/1805.06173.pdf)][[code](https://xueyangfu.github.io/projects/LPNet.html)]
### 2018
*Chen et. al. Gated Context Aggregation Network for Image Dehazing and Deraining. (2018 WACV) [[paper](https://arxiv.org/pdf/1811.08747.pdf)][[code](https://github.com/cddlyf/GCANet)]
* Pu, Jinchuan et al. Removing rain based on a Cycle Generative Adversarial Network. (2018 ICIEA) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8397790&tag=1)][code]
* Li, Xia et al. Recurrent Squeeze-and-Excitation Context Aggregation Net for Single Image Deraining. (2018 ECCV) [[paper](https://arxiv.org/pdf/1807.05698.pdf)][[code](https://xialipku.github.io/RESCAN/)]
* Fan, Zhiwen et al. Residual-Guide Feature Fusion Network for Single Image Deraining. (2018 ACM'MM) [[paper](https://arxiv.org/pdf/1804.07493.pdf)][code]
*Li, Guanbin et al. Non-locally Enhanced Encoder-Decoder Network for Single Image De-raining. (2018 ACM'MM) [[paper](https://arxiv.org/pdf/1808.01491.pdf)][[code](https://github.com/AlexHex7/NLEDN)]
*Pan, Jinshan et al. Learning Dual Convolutional Neural Networks for Low-Level Vision. (2018 CVPR) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8578422)][code]
* Qian, Rui et al. Attentive Generative Adversarial Network for Raindrop Removal from a Single Image. (2018 CVPR) (tips: this research focuses on reducing the effets form the adherent rain drops instead of rain streaks removal) [[paper](https://arxiv.org/pdf/1711.10098.pdf)][[code](https://github.com/rui1996/DeRaindrop)]
*Zhang, He et al. Density-aware Single Image De-raining using a Multi-stream Dense Network. (2018 CVPR) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8578177)][[code](https://github.com/hezhangsprinter/DID-MDN)]

* Du, Shuangli et al. Single image deraining via decorrelating the rain streaks and background scene in gradient domain. (2018 PR) [[paper](https://www.sciencedirect.com/science/article/pii/S0031320318300700)][code]
### 2017
* Zhang, He et al. Image De-raining Using a Conditional Generative Adversarial Network. (2017 Arxiv) [[paper]([[paper](https://www.sciencedirect.com/science/article/pii/S0031320318300700)][code])][[code](https://github.com/hezhangsprinter/ID-CGAN)] 
* Chang, Yi et al. Transformed Low-Rank Model for Line Pattern Noise Removal. (2017 ICCV) [[paper](http://openaccess.thecvf.com/content_iccv_2017/html/Chang_Transformed_Low-Rank_Model_ICCV_2017_paper.html)][code]
* JBO 

* Wei, Wei et al. Joint Bi-layer Optimization for Single-image Rain Streak Removal. (2017 ICCV) [paper][[paper](http://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Joint_Bi-Layer_Optimization_ICCV_2017_paper.html)]
* Gu, Shuhang et al. Joint Convolutional Analysis and Synthesis Sparse Representation for Single Image Layer Separation. (2017 ICCV) [[paper](http://openaccess.thecvf.com/content_iccv_2017/html/Gu_Joint_Convolutional_Analysis_ICCV_2017_paper.html)][code]
* Fu, Xueyang et al. Removing rain from single images via a deep detail network. (2017 CVPR) [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Fu_Removing_Rain_From_CVPR_2017_paper.pdf)] [[code](https://xueyangfu.github.io/projects/cvpr2017.html)]
* Yang, Wenhan et al. Deep joint rain detection and removal from a single image. (2017 CVPR) [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yang_Deep_Joint_Rain_CVPR_2017_paper.pdf)] [[code](http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)]
* Wang, Yinglong et al. A Hierarchical Approach for Rain or Snow Removing in a Single Color Image. (2017 TIP) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7934435)][code]
* Fu, Xueyang et al. Clearing the skies: A deep network architecture for single-image rain removal. (2017 TIP) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7893758)][[code](https://xueyangfu.github.io/projects/tip2017.html)]
### before 2017
* Li, Yu et al. Single Image Rain Streak Decomposition Using Layer Priors. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7934436)] [dataset]
* Luo, Yu et al. Removing rain from a single image via discriminative sparse coding. (2015 ICCV) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7410745)][[code](https://cs.nyu.edu/~deigen/rain/)]
* David, Eigen et al. Restoring An Image Taken Through a Window Covered with Dirt or Rain. (2013 ICCV) [[paper](https://cs.nyu.edu/~deigen/rain/)][[code](https://cs.nyu.edu/~deigen/rain/)]
* Kang, Liwei et al. Automatic Single-Image-Based Rain Streaks Removal via Image Decomposition. (2012 TIP) [[paper](https://www.ee.nthu.edu.tw/cwlin/Rain_Removal/tip_rain_removal_2011.pdf)][[web](https://www.ee.nthu.edu.tw/~cwlin/pub.htm)]
## 2.2 Video Based Deraining
### 2019
* Yang, Wenhan et al. D3R-Net: Dynamic Routing Residue Recurrent Network for Video Rain Removal. (2019 TIP) [[paper](http://www.icst.pku.edu.cn/struct/Pub%20Files/2019/ywh_tip19.pdf)][code]
### 2018
*Li, Minghan et al. Video Rain Streak Removal By Multiscale ConvolutionalSparse Coding. (2018 CVPR) [[paper](https://pan.baidu.com/s/1iiRr7ns8rD7sFmvRFcxcvw?errno=0&errmsg=Auth%20Login%20Sucess&&bduss=&ssnerror=0&traceid=)][[code](https://github.com/MinghanLi/MS-CSC-Rain-Streak-Removal)]
* Chen, Jie et al. Robust Video Content Alignment and Compensation for Rain Removal in a CNN Framework. (2018 CVPR) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8578756)][code][[web](https://github.com/hotndy/SPAC-SupplementaryMaterials)]
*Liu, Jiaying et al. Erase or Fill? Deep Joint Recurrent Rain Removal and Reconstruction in Videos. (2018 CVPR) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Erase_or_Fill_CVPR_2018_paper.pdf)][[code](https://github.com/flyywh/J4RNet-Deep-Video-Deraining-CVPR-2018)][[web](http://39.96.165.147/people/liujiaying.html)] 
### 2017
* Wei, Wei et al. Should We Encode Rain Streaks in Video as Deterministic or Stochastic? (2017 ICCV)  [[paper](http://openaccess.thecvf.com/content_iccv_2017/html/Wei_Should_We_Encode_ICCV_2017_paper.html)] [[code](https://github.com/wwzjer/RainRemoval_ICCV2017)][[web](https://github.com/wwzjer/RainRemoval_ICCV2017)]
* Jiang, Taixiang et al. A novel tensor-based video rain streaks removal approach via utilizing discriminatively intrinsic priors. (2017 CVPR) [[paper](http://openaccess.thecvf.com/content_cvpr_2017/html/Jiang_A_Novel_Tensor-Based_CVPR_2017_paper.html)][[code](https://github.com/TaiXiangJiang/FastDeRain)]
* Ren, Weilong et al. Video Desnowing and Deraining Based on Matrix Decomposition. (2017 CVPR) [[paper](http://openaccess.thecvf.com/content_cvpr_2017/html/Ren_Video_Desnowing_and_CVPR_2017_paper.html)][[code](http://openaccess.thecvf.com/content_cvpr_2017/html/Ren_Video_Desnowing_and_CVPR_2017_paper.html)][[web](http://openaccess.thecvf.com/content_cvpr_2017/html/Ren_Video_Desnowing_and_CVPR_2017_paper.html)]

### before 2017
* You, Shaodi et al. Adherent raindrop modeling, detectionand removal in video. (2016 TPAMI) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7299675)][code]
* Kim, JH et al. Video deraining and desnowing using temporal correlation and low-rank matrix completion. (2015 TIP) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7101234)][[code](http://mcl.korea.ac.kr/~jhkim/deraining/)][[web](http://mcl.korea.ac.kr/~jhkim/deraining/)]
