# Deep-Probabilistic-Scaling

![Example Image](38intro.png)

Deep Probabilistic Scaling (DPS) is an uncertainty quantification tool for the control of misclassification error in (binary) neural classification. The algorithm relies on probabilistic scaling, a branch of [order statistics](https://en.wikipedia.org/wiki/Order_statistic) for non-parametric inference with confidence bounds on the prediction. 

DPS is a direct application of [Scalable Classification](https://paperswithcode.com/paper/probabilistic-safety-regions-via-finite) to convolutional neural networks for (binary) classification:

<div style="text-align:center;">
    <img src="binary_CNN.png" width="600">
</div>

that is the predictor function of the network $\hat \varphi$ such that

<div style="text-align:center;">
    <img src="scalable_class.png" width="300">
</div>

This framework allows to define a special region $\mathcal{S}_\varepsilon$ such that the probability of observing a false negative is bounded by $\varepsilon$:

<div style="text-align:center;">
    <img src="safety_bound.png" width="250">
</div>

## Content of the Repository

This repository contains the code for the experiments to validate the DPS algorithm. 

We considered 6 benchmarks datasets, on which we defined a binary classification problem, as shown below:
<img width="322" alt="Schermata 2024-06-13 alle 10 31 43" src="https://github.com/AlbiCarle/Deep-Probabilistic-Scaling/assets/77918497/e346188d-058b-440f-9b47-e6d1cc6d1992">

As an example, MNIST data are also available in the ``data`` folder in the ``.npy`` format.

The following notebooks are available:

- ``get_pneumoniaMNIST_data.ipynb`` shows how to download and save data for pnemoniaMNIST by using [medmnist](https://pypi.org/project/medmnist/) python library

- ``DeepSC_NNtraining.ipynb``: contains the training of the convolutional models (3-layer CNNs) used for DPS. The models are also shared in the  ``models`` folder
  
- ``DeepSC_ProbScaling.ipynb``:  main code to implement DPS
  
- ``EvaluationMetricsPlot.ipynb``: computes the evaluation metrics and plots the results for all the considered datasets.

## References

DPS was implemented for a research paper presented to [COPA2024](https://www.copa-conference.com/) on the basis of the concept paper 

Carlevaro, Alberto, et al. "Probabilistic Safety Regions Via Finite Families of Scalable Classifiers." arXiv preprint arXiv:2309.04627 (2023).

The full paper will be available in the proceedings of the conference.
