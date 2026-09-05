# References

Full citations for the sources referenced throughout this documentation,
drawn from the thesis bibliography.

## Model architectures

- **[16]** Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung, Wai-Kin Wong,
  and Wang-chun Woo. *Convolutional LSTM network: A machine learning
  approach for precipitation nowcasting.* NeurIPS, 28:802–810, 2015.
  [arXiv:1506.04214](https://arxiv.org/abs/1506.04214)
- **[17]** Sepp Hochreiter and Jürgen Schmidhuber. *Long short-term memory.*
  Neural Computation, 9(8):1735–1780, 1997.
- **[5]** Jie Shi, Aleksej Cornelissen, and Siamak Mehrkanoon. *Integrating
  weather station data and radar for precipitation nowcasting: SmaAt-Fusion
  and SmaAt-Krige-GNet.* arXiv:2502.16116, 2025.
- **[18]** Olaf Ronneberger, Philipp Fischer, and Thomas Brox. *U-Net:
  Convolutional networks for biomedical image segmentation.* MICCAI,
  234–241, 2015. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
- **[6]** Zhihan Gao, Xingjian Shi, Hao Wang, Yi Zhu, Yuyang Wang, Mu Li, and
  Dit-Yan Yeung. *Earthformer: Exploring space-time transformers for earth
  system forecasting.* NeurIPS, 35:25390–25403, 2022.
  [arXiv:2207.05833](https://arxiv.org/abs/2207.05833)
- **[54]** Zhangyang Gao, Cheng Tan, Lirong Wu, and Stan Z. Li. *SimVP:
  Simpler yet better video prediction.* CVPR, 3170–3180, 2022.
  [arXiv:2206.05099](https://arxiv.org/abs/2206.05099)
- **[55]** Xi Ye and Guillaume-Alexandre Bilodeau. *VPTR: Efficient
  transformers for video prediction.* ICPR, 3492–3499, 2022.
  [arXiv:2203.15836](https://arxiv.org/abs/2203.15836)
- **[56]** Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell.
  *Simple and scalable predictive uncertainty estimation using deep
  ensembles.* NeurIPS, 30, 2017.
- **[52]** Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. *Sequence to
  sequence learning with neural networks.* NeurIPS, 27:3104–3112, 2014.
- **[53]** Sylwester Klocek et al. *MS-Nowcasting: Operational precipitation
  nowcasting with convolutional LSTMs at Microsoft Weather.* NeurIPS 2021
  Workshop on Tackling Climate Change with ML, 2021.

## Data sources and preprocessing

- **[37]** EUMETSAT. *Rapid Scan High Rate SEVIRI Level 1.5 image data –
  MSG.* [EUMETSAT Data Portal](https://user.eumetsat.int/catalogue/EO:EUM:DAT:MSG:MSG15-RSS),
  2024.
- **[39]** Zeqing Huang, Tongtiegang Zhao, Rongbiao Lai, Yu Tian, and Fang
  Yang. *A comprehensive implementation of the log, Box–Cox and log-sinh
  transformations for skewed and censored precipitation data.* Journal of
  Hydrology, 620:129347, 2023.
- **[40]** Chittaranjan Andrade. *Z scores, standard scores, and composite
  test scores explained.* Indian Journal of Psychological Medicine,
  43(6):555–557, 2021.
- **[41]** Tamás Frajka and Kenneth Zeger. *Downsampling dependent
  upsampling of images.* Signal Processing: Image Communication,
  19(3):257–265, 2004.
- **[42]** Haonan Chen, V. Chandrasekar, Robert Cifelli, and Pingping Xie.
  *A machine learning system for precipitation estimation using satellite
  and ground radar network observations.* IEEE TGRS, 58(2):982–994, 2020.
- **[45]** Connor Shorten and Taghi M. Khoshgoftaar. *A survey on image data
  augmentation for deep learning.* Journal of Big Data, 6(1):60, 2019.
- **[46]** Atharva Deshpande, Kaushik Gopalan, Jeet Shah, and Hrishikesh
  Simu. *A conditional generative adversarial network model for the
  Weather4cast 2024 challenge.* arXiv:2412.00451, 2024.
- **[47]** N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer.
  *SMOTE: Synthetic minority over-sampling technique.* Journal of
  Artificial Intelligence Research, 16:321–357, 2002.
- **[48]** Yuan Cao, Lei Chen, Danchen Zhang, Leiming Ma, and Hongming Shan.
  *Hybrid weighting loss for precipitation nowcasting from radar images.*
  IEEE ICASSP, 2022.
- **[50]** Mariana Oliveira, Luís Torgo, and Vítor Santos Costa. *Evaluation
  procedures for forecasting with spatiotemporal data.* Mathematics,
  9(6):691, 2021.

## Evaluation metrics

- **[58]** Trevor Hastie, Robert Tibshirani, and Jerome Friedman. *The
  Elements of Statistical Learning: Data Mining, Inference, and
  Prediction.* Springer, 2009.
- **[59]** T. Chai and R. R. Draxler. *Root mean square error (RMSE) or mean
  absolute error (MAE)? – Arguments against avoiding RMSE in the
  literature.* Geoscientific Model Development, 7(3):1247–1250, 2014.
- **[60]** Ian Goodfellow, Yoshua Bengio, and Aaron Courville. *Deep
  Learning.* MIT Press, 2016.
- **[61]** Daniel S. Wilks. *Statistical Methods in the Atmospheric
  Sciences.* Academic Press, 3rd edition, 2011.
- **[62]** Chung-Chieh Wang. *On the calculation and correction of equitable
  threat score for model quantitative precipitation forecasts for small
  verification areas: The example of Taiwan.* Weather and Forecasting,
  29(4):788–798, 2014.
- **[63]** Q. Huynh-Thu and M. Ghanbari. *Scope of validity of PSNR in
  image/video quality assessment.* Electronics Letters, 44(13):800–801,
  2008.
- **[64]** Elizabeth E. Ebert. *Fuzzy verification of high-resolution
  gridded forecasts: A review and proposed framework.* Meteorological
  Applications, 15(1):51–64, 2008. *(source of the "double penalty problem"
  discussed in [Results & Findings](results.md))*

## Baselines, limitations, and future work

- **[12]** Seppo Pulkkinen, Daniele Nerini, Andrés A. Pérez Hortal, Carlos
  Velasco-Forero, Alan Seed, Urs Germann, and Loris Foresti. *pySTEPS: An
  open-source Python library for probabilistic precipitation nowcasting
  (v1.0).* Geoscientific Model Development, 12(10):4185–4219, 2019.
- **[3]** Urs Germann and Isztar Zawadzki. *Scale-dependence of the
  predictability of precipitation from continental radar images. Part I:
  Description of the methodology.* Monthly Weather Review,
  130(12):2859–2873, 2002.
- **[20]** Yan Ji, Bing Gong, Michael Langguth, Amirpasha Mozaffari, and
  Xiefei Zhi. *CL-GAN: A generative adversarial network (GAN)-based video
  prediction model for precipitation nowcasting.* Geoscientific Model
  Development, 16:2737–2752, 2023.
- **[22]** Jussi Leinonen, Ulrich Hamann, Daniele Nerini, Urs Germann, and
  Gabriele Franch. *Latent diffusion models for generative precipitation
  nowcasting with accurate uncertainty quantification.* arXiv:2304.12891,
  2023.
- **[65]** Hong Wang, Dehui Chen, Jinfang Yin, Daosheng Xu, Guangfeng Dai,
  and Luwen Chen. *An improvement of convective precipitation nowcasting
  through lightning data dynamic nudging in a cloud-resolving scale
  forecasting system.* Atmospheric Research, 242:104994, 2020.
- **[66]** Fuzhen Zhuang, Zhiyuan Qi, Keyu Duan, Dongbo Xi, Yongchun Zhu,
  Hengshu Zhu, Hui Xiong, and Qing He. *A comprehensive survey on transfer
  learning.* Proceedings of the IEEE, 109(1):43–76, 2021.

!!! note
    Bracketed numbers `[N]` match the citation numbering used in the
    thesis itself, so they can be cross-referenced directly against the
    full Bibliography in the submitted document.
