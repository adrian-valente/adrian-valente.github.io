---
layout: page
title: Research
---
## Population dynamics in the brain and in artificial neural networks
How do cortical circuits implement cognitive capacities, like working memory or decision making? When looking at recordings of behaving animals, it is hard to make sense of the role of individual neurons by looking at the activity of single cells. However, when looking at assemblies of neurons, the activity seems to vary along low-dimensional manifolds of the neural state space, and one can identify axes that seem to encode computational variables [1]. Similarly, when training artifical recurrent neural networks to perform cognitive tasks, one can interpret their activity in suitable low-dimensional subspaces (Sussillo & Barak, 2012). 

This seems to suggest a common computational mechanism for both cortical circuits and artificial RNNs: those can be seen dynamical systems that can be trained to approximate low-dimensional dynamics that implement computations based on attractor states and manifolds, like fixed points (for memory encoding), line or ring attractors (to integrate 1d variables or directions), or limit cycles (to produce oscillations).

Can we model this emergent property of neural networks, and what does observation of the activity patterns tell us about what is implemented in a network? These remain hard questions, even with a fully accessible model like a trained RNN, due to the complexity of this non-linear system. However, one way to obtain an emergent low-dimensional activity with computational capabilities in a neural network has been demonstrated in (Mastroguiseppe & Ostojic, 2019) by considering RNNs whose connectivity matrix is low-rank. 

My research focuses on exploiting this framework to uncover a mechanistic understanding of RNNs trained to implement cognitive tasks. 

Bibliography:

[1] Mante, Valerio, David Sussillo, Krishna V. Shenoy, and William T. Newsome. “Context-Dependent Computation by Recurrent Dynamics in Prefrontal Cortex.” Nature 503, no. 7474 (November 2013): 78–84. https://doi.org/10.1038/nature12742.

[2] Sussillo, David, and Omri Barak. “Opening the Black Box: Low-Dimensional Dynamics in High-Dimensional Recurrent Neural Networks.” Neural Computation 25, no. 3 (December 28, 2012): 626–49. https://doi.org/10.1162/NECO_a_00409.

[3] Mastrogiuseppe, Francesca, and Srdjan Ostojic. “Linking Connectivity, Dynamics, and Computations in Low-Rank Recurrent Neural Networks.” Neuron 99, no. 3 (August 8, 2018): 609-623.e29. https://doi.org/10.1016/j.neuron.2018.07.003.
