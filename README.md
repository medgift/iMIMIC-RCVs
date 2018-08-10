# Regression Concept Vectors for Bidirectional Explanations in Histopathology (iMIMIC at MICCAI 2018)
Mara Graziani, Vincent Andrearczyk, Henning Muller.

To be presented at Interpretability of Machine Intelligence in Medical Image Computing at MICCAI 2018.

Paper Preview:

This repository contains the code for implementing Regression Concept Vectors and Bidirectional Relevance scores, which were used to obtain the results presented in the paper.

## Dependencies
This code runs in Python >= 2.7.
Keras >= 2.1 and Tensorflow backend are required.

To install the list of dependencies run:

    pip install requirements.txt

## Usage
This repository contains a jupyter notebook and the link to the necessary data and trained models to replicate the results. Please open the notebook RCV_notebook.ipynb for more information.

## Results

#### Correlation Analysis
As a prior analysis, we compute the Pearson product-moment correlation coefficient between the concept measures and the network prediction for a set of patches. More information about the Correlation Analysis and the concept measures used can be found in the paper.


|     | correlation | ASM | eccentricity | Euler | area | contrast |
| --- | ----------- | --- | ------------ | ----- | ---- | -------- |
r |-0.2285 | -0.1869 | -0.1460 | 0.1534| 0.2820 | 0.4119|
p-value |0.001 |0.001 | 0.01 |0.001 |0.001 |0.001|

#### Are we learning the concepts?

The performance of the linear regression was
computed for all the patches over
multiple reruns to check if the network is learning the concepts and in which layers. 
The learning of the concepts across layers is linked to the size of the
receptive field of the neurons and the increasing complexity of the sought patterns. 
Hence, more abstract concepts, potentially useful in other applications, can be learned and analyzed in deep layers of the network.

<p align="center">
    <img src="results/featslearningdef.png" width=700px>
</p>

## Further applications
We applied RCVs to eye data. See this repository for more details.

## Relevant Research
 * Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (https://arxiv.org/abs/1711.11279; https://github.com/tensorflow/tcav/blob/master/README.md)

## Credits
RCVs were computed by extending the keras-vis library: https://github.com/raghakot/keras-vis.
Staining normalization was performed thanks to the StainTools library: https://github.com/Peter554/StainTools.git.
