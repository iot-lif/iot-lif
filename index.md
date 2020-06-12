---
layout: home 
---

The purpose of this project is to explore the usage of Spiking Neural Networks and
an approach for training them (Deep Continuous Local Learning - DECOLLE) in
Software Defined Radios for the task of modulation classification.

The project is carried out by Owen Jow and Simon Kaufmann at the University of California,
San Diego during Spring Quarter 2020 for the course CSE 145/237D.

<iframe width="90%" height="400px" src="https://www.youtube.com/embed/ZPpojEGsSfE" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Background

Radio communication plays an important role in modern day life and
shapes the way people are able to communicate with each other through
mobile devices. A growing number of communication services (like the 5G
mobile phone network) need to share a very limited frequency spectrum.
The ability to share the same frequency range for different services
could therefore be useful in the future, but requires being able to
classify and recognise received radio signals by the service they belong
to. In this project, we aim to implement an efficient and low-power
computing system to classify radio signals. Our method will be based on
a learning system inspired by biological neurons and will be evaluated
using RadioML, a publicly available dataset of radio signals.

## Goal

Based on the [DCLL library](https://github.com/nmi-lab/dcll) implementation
for spiking neural networks and deep continuous local learning, we are optimizing
a neural network to classify the radio signal data of the [RadioML Dataset](https://www.deepsig.io/datasets).

![Modulations in RadioML dataset](/assets/img/dataset_time.png)

The trained network is then quantized using the [Brevitas](https://github.com/Xilinx/brevitas) quantization framework
to prepare it for efficient implementation on an FPGA.

Previous work on modulation classification used classic convolutional neural
networks for the task. Due to less developed and sophisticated training methods
available for SNNs, we do not anticipate to achieve better accuracy on the dataset,
but we hope to lay groundwork for an efficient low-power implementation that
will require fewer resources than conventional neural networks.
