# Automatic Line Art Colorization

## Introduction
This repository implements converting line arts into color images automatically. Of course, we can train the neural network to deal with line art only. However, in the application system, we also need to colorize the line art with designated color in advance. There are many types of colorization with respect to hint.

- No hint
  - Colorization without hint
  - Input: Line art only
  
- Atari
  - Colorization with hint which is line of desired color in the specific region (ex. PaintsChainer)
  - Input: Line art and atari
  
- Tag
  - Colorzation with hint which is tag (ex. Tag2Pix)
  - Input: Line art and tag
  
- Reference
  - Colorization with hint which is reference image (ex. style2paints V1)
  - Input: Line art and reference image
  
## Line extraction method
There are many variations in line extraction methods, such as XDoG or SketchKeras. But, when trained on only one type of line art, trained model comes to overfit to this type of line art and this model doesn't fully colorize another type of line art. Therefore, like Tag2Pix, I use various kinds of line art as the input of neural network.

I use three types of line art below.

- XDoG
  - Line extraction using two Gaussian distributions difference to standard deviations
  
- SketchKeras
  - Line extraction using UNet. Lines obtained by SketchKeras are like pencil drawings.
  
- Sketch Simplification
  - Line extraction using Fully-Convolutional Networks. Lines obtained by Sketch Simplification are like digital drawings.

Examples obtained by these line extraction methods are as follows.  

![](https://github.com/SerialLain3170/Colorization/blob/master/Data/lineart.png)

Moreover, I consider three types of data augmenation to line arts in order to avoid overfitting.

- Adding intensity
- Randomly morphology transformation to deal with various thicks of lines
- Randomly RGB values of lines to deal with various depths of lines

## Experiment without hint

### Motivation
First of all, I need to confirm that method based on neural networks can colorize without hint precisely and diversely. The training of mapping from line arts to color images is difficult because variations in color exist. Therefore, without hint, I think the neural networks come to colorize single color in any regions. To avoid falling into local minimum, I try adversarial loss in addition to the content loss because adversarial learning trains neural network of colorization to match data distribution precisely.

### Methods
- [x] pix2pix
- [x] pix2pixHD
- [X] bicyclegan

### Results
- pix2pix & pix2pixHD

![](./Data/nohint_comparison.png)

- bicyclegan

![](https://github.com/SerialLain3170/Colorization/blob/master/nohint_bicyclegan/data/result1.png)

## Experiment with atari

### Motivation
Watching results of experiments above, even with using adversarial loss, it seems that neural network falls local minimum. Some degrees of variations in color may exist, neural networks seem to learn to colorize single color to any regions in single character. I find it difficult to train mapping from line art to color image without hint. Therefore, I consider taking the hint, atari, as input of neural network.

### Methods
- [x] userhint
- [x] whitebox
- [ ] gaugan

### Results
- userhint
![here](https://github.com/SerialLain3170/Colorization/blob/master/atari_userhint/data/result2.png)

- whitebox
![](https://github.com/SerialLain3170/Colorization/blob/master/atari_whitebox/data/result2.png)

## Experiment with reference

### Motivation
I also consider taking the hint, reference, as input of neural network. First of all, I had tried to implement style2paints V1. However, I had difficulities producing the reproduction of results because training came to collapse. Then, I decide to seek for a substitute for style2paints V1.

### Methods
- [x] adain
- [x] scft
- [ ] video

### Result
- adain
![here](https://github.com/SerialLain3170/Colorization/blob/master/reference_adain/data/res1.png)

- scft
![](https://github.com/SerialLain3170/Colorization/blob/master/reference_scft/data/result2.png)

- video
![](https://github.com/SerialLain3170/Colorization/blob/master/reference_video/data/never_color1.gif)
![](https://github.com/SerialLain3170/Colorization/blob/master/reference_video/data/sakura1_color1.gif)
![](https://github.com/SerialLain3170/Colorization/blob/master/reference_video/data/rayearth1_color1.gif)
