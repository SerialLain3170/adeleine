# Automatic Line Art Colorization

## Introduction
This repository implements converting line arts into color images automatically. Of course, we can train the neural network to deal with line art only. However, in the application system, we also need to colorize the line art with designated color in advance. There are many types of colorization with respect to hint.

- No hint
  - Colorization without hint
  - Input: Line art only
  
- Atari
  - Colorization with hint which is line of desired color in the specific region
  - Input: Line art and atari
  
- Tag
  - Colorzation with hint which is tag
  - Input: Line art and tag
  
- Reference
  - Colorization with hint which is reference image 
  - Input: Line art and reference image

## Quick Results
### pix2pix
[Paper](https://arxiv.org/pdf/1611.07004.pdf)  

![pix2pix](https://github.com/SerialLain3170/Line-to-Color/blob/master/pix2pix/result.png)
![pix2pix2](https://github.com/SerialLain3170/Line-to-Color/blob/master/pix2pix/result2.png)

### pix2pix + wasserstein metrics
[Paper](https://arxiv.org/pdf/1808.03240v1.pdf)  

![here](https://github.com/SerialLain3170/Line-to-Color/blob/master/Wasserstein/result.png)

### pix2pixHD
[Paper](https://arxiv.org/pdf/1711.11585.pdf)

![here](https://github.com/SerialLain3170/Line-to-Color/blob/master/pix2pixHD/visualize_125.png)

### UserHint
今までは、線画のみを与えて着色を行っていた。ここでは、カラーヒントを与えて着色を行う

![here](https://github.com/SerialLain3170/Line-to-Color/blob/master/UserHint/example.png)

### style2paints
![here](https://github.com/SerialLain3170/Colorization/blob/master/style2paints/Result.png)
