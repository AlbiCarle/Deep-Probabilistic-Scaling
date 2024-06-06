# Deep-Probabilistic-Scaling

![Example Image](38intro.png)

Deep Probabilistic Scaling is an uncertainty quantification tool for the control of misclassification error in (binary) neural classification. The algorithm relies on probabilistic scaling, a branch of [order statistics](https://en.wikipedia.org/wiki/Order_statistic) for non-parametric inference with confidence bounds on the prediction. 

Deep Probabilistic Scaling is a direct application of [Scalable Classification](https://paperswithcode.com/paper/probabilistic-safety-regions-via-finite) to convolutional neural networks for (binary) classification:

<div style="text-align:center;">
    <img src="binary_CNN.png" width="600">
</div>

that is the predictor function of the network such that

<div style="text-align:center;">
    <img src="scalable_class.png" width="400">
</div>

This framework allows to define a special region $\mathcal{S}_\varepsilon$ such that the probability of observing an event tha


