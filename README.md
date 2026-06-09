# Explainable EfficientNet for Grapevine Pierce’s Disease Stage Classification
Pierce’s disease is a bacterial disease affecting grapevines, induced by Xylella fastidiosa.
Its symptoms include leaf scorching, wilting, and ultimately it becomes the cause of
plant death. An accurate classification of its progression is important in precision
agriculture but current methods based on manual diagnosis are subjective, imprecise,
and inefficient. To the best of our knowledge, no previous study addressed multi-stage
classification of Pierce’s Disease using Deep Learning or Computer Vision. This paper
proposes PierceStageNet, a transfer learning-based explainable model for classifying four
stages of grapevine Pierce’s Disease from single leaf images. PierceStageNet uses an
EfficientNet-B0 backbone network, along with a Squeeze-and-Excitation layer and
classification layers. The explainability of the model is facilitated through visualization
of feature maps and Gradient-weighted Class Activation Map (Grad-CAM). The model
is evaluated on a dataset containing 4,004 grapevine leaf images, achieving accuracies of
0.9751, 0.9754, 0.9750, and 0.9749 for Accuracy, Precision, Recall, and F1-Score,
respectively. These results outperform those of the baseline EfficientNet-B0 model
under identical experimental setup.




<img width="1426" height="997" alt="fig4" src="https://github.com/user-attachments/assets/c0c14e50-94ec-49ad-8c6f-a598cf477edf" />

<img width="1920" height="1353" alt="fig3" src="https://github.com/user-attachments/assets/eea89442-32b9-4bfb-a801-2a68b6571347" />

