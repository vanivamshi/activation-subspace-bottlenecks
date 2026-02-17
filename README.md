<h1 align="center"> 🔍 Activation subspace bottlenecks 🔍 </h1>
<p align="center"> <b>Interpreting and Steering State-Space Models via Activation Subspace Bottlenecks</b> (<a href="">Mohan et al. arXiv 2026</a>). 
</p>

> State-space models (SSMs) have emerged as an efficient strategy for building powerful language models, avoiding the quadratic complexity of computing attention in transformers. Despite their promise, the interpretability and steerability of modern SSMs remain relatively underexplored. We take a major step in this direction by identifying activation subspace bottlenecks in the Mamba family of SSM models using tools from mechanistic interpretability. We then introduce a test-time steering intervention that simply multiplies the activations of the identified bottlenecks by a scalar. Across 5 SSMs and 6 diverse benchmarks, this intervention improves performance by an average of 8.27%, without requiring any task-specific tuning. Finally, we validate that the identified bottlenecks are indeed hindering performance by modifying them to yield an architecture we call Stable-Mamba, which achieves long-context performance gains when retrained from scratch.

## Reproducing experiments
See readme file within each folder for instructions on reproducing the experiments.
