# Automatic Line Art Colorization

## Introduction
This repository implements converting line arts into color images automatically. In addition to training the neural network with line art only, this repository is able to colorize the line art with several types of hint in advance. There are mainly for types of hints.

- No hint
  - description: Colorization without hint
  - input: Line art only
  
- Atari
  - description: Colorization with hint that includes some lines in desired color (ex. PaintsChainer)
  - input: Line art and atari
  
- Tag
  - description: Colorzation with tag (ex. Tag2Pix)
  - input: Line art and tag
  
- Reference
  - description: Colorization with reference images (ex. style2paints V1)
  - input: Line art and reference image
  
## Line extraction method
There are many variations in line extraction methods, such as XDoG or SketchKeras. However, if we train the model on only one type of line art, trained model comes to overfit and the model are not able to colorize another type of line art adequately. Therefore, like Tag2Pix, I use various kinds of line art as the input of neural network.

I use mainly three types of line art.

- XDoG
  - Line extraction using two Gaussian distributions difference to standard deviations
  
- SketchKeras
  - Line extraction using UNet. Lines obtained by SketchKeras are like pencil drawings.
  
- Sketch Simplification
  - Line extraction using Fully-Convolutional Networks. Lines obtained by Sketch Simplification are like digital drawings.

Examples obtained by these line extraction methods are as follows.  

![](https://github.com/SerialLain3170/Colorization/blob/master/Data/lineart.png)

Moreover, I add three types of data augmenation to line arts in order to avoid overfitting.

- Adding intensity
- Randomly morphology transformation to deal with various thicks of lines
- Randomly RGB values of lines to deal with various depths of lines

## Experiment without hint

### Motivation
First of all, I need to confirm that method based on neural networks can colorize without hint precisely and diversely. The training of mapping from line arts to color images is difficult because of variations in color. Therefore, I hypothesize that the neural networks trained without hint come to colorize single color in any regions. To avoid this, I try adversarial loss in addition to the content loss because adversarial learning enables neural networrks to match data distribution adequately.

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
Considering the application systems of colorization, we need to colorize with designated color. Therefore, I try some methods that take the hint, atari, as input of neural network.

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
