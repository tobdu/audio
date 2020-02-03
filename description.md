In crowd-sourced music tag datasets [2,13], most of the
tags are false(0) for most of the clips, which makes accuracy or mean square error inappropriate as a measure.
Therefore we use the Area Under an ROC (Receiver Operating Characteristic) Curve abbreviated as AUC. This measure has two advantages. It is robust to unbalanced datasets
and it provides a simple statistical summary of the performance in a single value. It is worth noting that a random
guess is expected to score an AUC of 0.5 while a perfect
classification 1.0, i.e., the effective range of AUC spans
between [0.5, 1.0].

https://arxiv.org/pdf/1606.00298.pdf