# Deep Image Homography Estimation

This work presents a summary and an implementation of [Deep Image Homography Estimation](https://arxiv.org/abs/1606.03798) by
Daniel DeTone, Tomasz Malisiewicz and Andrew Rabinovich pressented as Final Project in the course Computer Vision I.

Centro de Investigación en Matemáticas, A.C.

> Rodrigo Castillo González

## I. Introduction

Estimating a 2D homography (projective transformation) from a pair of images is a fundamental task in computer vision.
Specially in scenarios such as: rotation only movements, planar scenes and scenes in which objects are very far from the viewer.

The traditional homography estimation pipeline is composed of two stages: corner estimation and robust homography estimation. Where
robustness is introduced in two ways:

1. Corner detection: return a large and over-complete set of points.
2. Homography estimation: use of RANSAC or robustification of the squared loss function. \

The objective in this work is to construct a deep learning model with the ability to learn the entire homography estimation pipeline,
and the motivation for the network architechture is that currently, Convolutional Neural Networks (CNNs) set the state-of-the-art
in tasks such as image processing, segmentation and classification. Additionally, CNNs present promising results for dense geometric
computer vision tasks (optical flow and depth estimation).

## II. Method

## II. Methods

### A) The 4-point homography parametrization
The 4-point parameterization has been used in traditional homography estimation methods. Two views of a plane are related by a homography:

```math
\sqrt{3}
```

$$\boldsymbol{p} ' \ \sim \ H \boldsymbol{p}$$