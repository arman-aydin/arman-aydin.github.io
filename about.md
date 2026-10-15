---
layout: about
image: assets/img/Block_S_1_color_red_128.png
description: >
  Personal website of Arman Aydin, includes projects, resume, blog posts, and short
  description of his endeavors both academic and personal.
hide_description: true
---

# About Me

<!-- author -->
<!-- This should have the short blurb about me -->


1. this list will be replaced by the toc
{:toc .large-only}

## Education

**Stanford University**

  - **Master of Science in Computer Science** (Jan 2024 – Jun 2025)
    - *Relevant Coursework:* Machine Learning, Reinforcement Learning, Computer Vision, Natural Language Processing, Machine Learning with Graphs, Statistical Inference, Linear Dynamical Systems, Systems for Machine Learning

  - **Bachelor of Science in Computer Science** (Sep 2020 – Jun 2024)
    - *Relevant Coursework:* Data Structures and Algorithms, Parallel Computing, Operating Systems, Computer Networking, Compilers, Database Systems, Embedded Systems, Compiler Optimizations, Cryptography, Computer and Network Security

## Work Experience

**Software Engineer II**, *Box Inc.* | (Jul 2025 – Present)

> Technologies: Python, PyTorch, Docker, Kubernetes, Kedro, Prefect, Braintrust

- Architected synthetic data generation pipelines using Kedro, modeling open-source datasets as graphs; designed graph-traversal based algorithms (modeled on the MuSiQue framework) to programmatically construct complex, multi-hop reasoning benchmarks for enterprise-specific contexts.
- Designed and deployed an online grading pipeline modeled after Anthropic’s CLIO pipeline for emergent hierarchical clustering of user activity; for offline grading, utilized Braintrust for trajectory tracing and integrated automated stability metrics using semantic entropy and self-consistency variance.
- Engineered a high-throughput automated evaluation infrastructure integrated directly into the CI/CD pipeline; orchestrated containerized batch workloads to execute 10,000+ daily test cases, enforcing empirical quality baselines critical to the General Availability (GA) launch of the Box Agents platform (scaling to 1M+ users).

**Research Assistant**, *Stanford Vision and Learning Lab (SVL)* | (Mar 2023 – Sep 2024)

> Technologies: NVIDIA IsaacSim, PyTorch, Autodesk 3ds Max, Dask, DVC

- Contributed on the convex mesh data pipeline for BEHAVIOR-1K, a benchmark for evaluating Embodied AI solutions through interactive household activities with physics simulations supervised by Dr. Fei-Fei Li.
- Scripting in Autodesk 3ds Max, integrating CoACD performance metrics, where achieved up to 39% speed-up in internal sanity check scripts for integrity checks of physics simulations.
- Developed teleoperation features for OmniGibson, Stanford-developed robot/environment simulation platform built on top of NVIDIA’s IsaacSim, designing a VR-based trajectory collection pipeline to capture high-fidelity human demonstrations for transfer learning and offline reinforcement learning (RL) frameworks.

**CURIS Research Assistant**, *Stanford NLP Group* | (Jun 2022 – Sep 2022)

> Technologies: PyTorch, TensorFlow, Numpy, SQL, Bash Script, JavaScript, Pytest

- Implemented Named Entity Recognition (NER) language models for low-resource languages Kazakh (95% F1-score) and Uyghur (77% F1-score) by leveraging cross-lingual syntactic similarities with high-resource languages like Turkish through transfer learning (supervised by Dr. John Bauer and Dr. Chris Manning).
- Enhanced the Stanford-developed Stanza natural language analysis package, achieving an 11% increase in parsing speed for nested NER tags, and added generalized transliteration across scripts and word span correction features.

## Publications & Selected Projects {#projects}

### Publications

**BEHAVIOR Vision Suite: Customizable Dataset Generation via Simulation** ([Paper](https://arxiv.org/abs/2405.09546), [Website](https://behavior-vision-suite.github.io/))
*Presented in CVPR 2024*

**BEHAVIOR-1K: A Benchmark for Embodied AI with 1,000 Everyday Activities and Realistic Simulation** ([Paper](https://arxiv.org/abs/2403.09227), [Website](https://behavior.stanford.edu/))
*Presented in CoRL 2023*

### Selected Projects

**vllm-lite: Lightweight vLLM Inference Engine**

> Technologies: PyTorch, Triton, PagedAttention, Continuous Batching

- Built a high-performance LLM inference engine featuring a continuous batching scheduler with recomputation preemption and a custom, fused Triton PagedAttention decode kernel to dynamically manage non-contiguous KV cache blocks.
- Integrated FlashAttention for accelerated prefill computation and consolidated concurrent decodes into unified forward passes, achieving a 4.25x boost in generation throughput and a 95% reduction in time-to-first-token (TTFT) latency.

**MixKVQ: Mixed-Precision KV Cache**

> Technologies: Triton, PyTorch, INT4/INT8 Bit-Packing

- Implemented the MixKVQ architecture by building a calibration pipeline to extract outlier scales, coupled with a custom Triton packing kernel that routes critical outliers to BF16 while quantizing non-outliers to INT4/INT8, achieving a 19.5x execution speedup.
- Programmed a dequantizing FlashAttention kernel that executes on-the-fly dequantization directly in SRAM, reducing overall KV cache VRAM footprints by 40% (and isolated key caches by 64%) while matching full-precision logits to a 1e-2 tolerance.

**deepGCRL-Bench: Decoupled Multi-GPU Post-Training for Goal-Conditioned Diffusion Policies** ([GitHub](https://github.com/arman-aydin/deepGCRL-Bench), [Blog Post](/2026-10-14-scaling-goal-conditioned-rl-pipelined-disaggregated/))

> Technologies: PyTorch, CUDA IPC, Ray, FSDP2, Diffusion Transformers, Distributed Systems

- Architected a 4-stage evolutionary post-training pipeline for 232M-parameter Diffusion Transformer (DiT) policies, eliminating lockstep idling and scaling decentralized sampling to 24.5M transitions/sec.
- Eliminated host-memory serialization bottlenecks in Ray Plasma (>3.1s sync) by engineering peer-to-peer CUDA IPC streams, reducing parameter synchronization latency to 10.63 ms (300x acceleration).
- Implemented dual-stream concurrency with double-buffered ping-pong parameter weights, achieving 100% latency hiding and an atomic 3.8 µs zero-bubble parameter handoff between trainer and rollout engines.

## Teaching

**Teaching Assistant for CS 107**, *Stanford Computer Science Department* | (Sep 2023 - Jun 2025)

> Technologies: C, x86 Assembly, GNU Debugger (GDB), Valgrind, GNU Compiler Collection (GCC)

- Led hands-on discussion sections and held weekly office hours for a core systems programming class.
- Topics covered include: C programming, data representation, machine-level x86-64 assembly, memory organization and custom allocation, and systems optimization.

**Peer Tutor for Computer Science**, *Stanford Center for Teaching and Learning* | (Mar 2021 - Jun 2023)

- Provided one-on-one academic tutoring for core Computer Science courses (CS106A/B, CS103, CS107, CS109, CS111, CS161).

## Technical Skills

- **Languages:** Python, C, C++, Rust, JavaScript, TypeScript, SQL, Julia
- **Frameworks:** PyTorch, Triton, Node.js, Docker, Kubernetes, Scikit-learn, HuggingFace, vLLM, AWS

<!-- 
## Features

{% include features.md %}


## Download

{% include table.md %} -->

[jekyll]: https://jekyllrb.com

[blog]: /
[portfolio]: https://hydejack.com/examples/
[resume]: https://hydejack.com/resume/
[download]: https://hydejack.com/download/
[welcome]: https://hydejack.com/
[forms]: https://hydejack.com/forms-by-example/

[features]: #features
[news]: #build-an-audience
[syntax]: #syntax-highlighting
[latex]: #beautiful-math
[dark]: https://hydejack.com/blog/hydejack/2018-09-01-introducing-dark-mode/
[search]: https://hydejack.com/#_search-input
[grid]: https://hydejack.com/blog/hydejack/

[lic]: LICENSE.md
[pro]: licenses/PRO.md
[docs]: docs/README.md
[ofln]: docs/advanced.md#enabling-offline-support
[math]: docs/writing.md#adding-math

[kit]: https://github.com/hydecorp/hydejack-starter-kit/releases
[src]: https://github.com/hydecorp/hydejack
[gem]: https://rubygems.org/gems/jekyll-theme-hydejack
[buy]: https://gum.co/nuOluY

[gpss]: https://developers.google.com/speed/pagespeed/insights/?url=https%3A%2F%2Fhydejack.com%2Fdocs%2F
[rouge]: http://rouge.jneen.net
[katex]: https://khan.github.io/KaTeX/
[mathjax]: https://www.mathjax.org/
