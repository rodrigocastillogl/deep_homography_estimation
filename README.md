# Deep Image Homography Estimation

This work presents a summary and an implementation of [Deep Image Homography Estimation](https://arxiv.org/abs/1606.03798) by
Daniel DeTone, Tomasz Malisiewicz and Andrew Rabinovich pressented as Final Project in the course Computer Vision I.

Centro de Investigación en Matemáticas, A.C.

> Rodrigo Castillo González

## Introduction

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

## Methods

### The 4-point homography parametrization
The 4-point parameterization has been used in traditional homography estimation methods. Two views of a plane are related by a homography:

$$\boldsymbol{p} ' \ \sim \ H \boldsymbol{p}$$

$$\begin{bmatrix}
    u' \\
    v' \\
    1
\end{bmatrix}
\sim
\begin{bmatrix}
    H_{11} & H_{12} & H_{13} \\
    H_{21} & H_{22} & H_{23} \\
    H_{31} & H_{32} & H_{33} \\
\end{bmatrix}
\begin{bmatrix}
    u \\
    v \\
    1
\end{bmatrix}$$

Suppose $H_{33} = 1$, we want to find the $8$ unknown entries of $H$ from correspondences in the plane. We can write $2$ equations given one correspondence, these are

$$\begin{matrix}
    -uH_{11} & -vH_{12} & -H_{13} & & & & + \ (u' u) H_{31} & + \ (u' v) H_{32} & = \ - u'\\
    \ & & & -uH_{21} & -vH_{22} & -H_{23} & + \ (v' u)H_{31} & + \ (v' v)H_{32} & = \ - v'
\end{matrix}$$

so we need $4$ point correspondences to get a non homogeneous $8 \times 8$ linear system.

## Data generation

Training deep convolutional networks from scratch requires a large amount of data. To meet this requirement, we generate labeled
training examples by applying random projective transformations to a large dataset of natural images. Following the steps below to
generate a single training sample:

<p align="center">
<img src = "https://github.com/rodrigocastillogl/deep_homography_estimation/blob/master/imgs/data_generation.png" width = 50% height = 50%>
</p>

Source: [(DeTone, Malisiewicz and Rabinovich, 2016)](https://arxiv.org/abs/1606.03798)

Images resized to $320 \times 240$ and converted to grayscale.