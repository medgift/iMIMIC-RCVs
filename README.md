# Regression Concept Vectors for Bidirectional Explanations in Histopathology (iMIMIC at MICCAI 2018)
Mara Graziani, Vincent Andrearczyk, Henning Muller.

To be presented at Interpretability of Machine Intelligence in Medical Image Computing at MICCAI 2018.

Research Gate Preview:

This repository contains the code for implementing Regression Concept Vectors and Bidirectional Relevance scores, which were used to obtain the results presented in the paper.

## Dependencies
This code runs in Python >= 2.7.
Keras >= 2.1 and Tensorflow backend are required.

To install the list of dependencies run:

    pip install requirements.txt

## Usage
This repository contains a jupyter notebook and the link to the necessary data and trained models to replicate the results. Please open the notebook RCV_notebook.ipynb for more information.

## Results

## Further applications
We applied RCVs to eye data. See this repository for more details.

## Relevant Research
Testing with Concept Activation Vectors (TCAV)

## Credits
RCVs were computed by extending the keras-vis library: .
Staining normalization was performed thanks to the nanan library: .
