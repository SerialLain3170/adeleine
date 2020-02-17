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
  
## Experiment without hint

### Motivation
First of all, I need to confirm that method based on neural networks can colorize without hint precisely and diversely. The training of mapping from line arts to color images is difficult because variations in color exist. Therefore, without hint, I think the neural networks come to colorize single color in any regions. To avoid falling into local minimum, I try adversarial loss in addition to the content loss because adversarial learning trains neural network of colorization to match data distribution precisely.

### Methods
- [x] pix2pix
- [x] pix2pix-gp (pix2pix + zero-centered gradient penalty)
- [x] pix2pixHD

### Results
- pix2pix

![pix2pix](https://github.com/SerialLain3170/Line-to-Color/blob/master/pix2pix/result.png)


- pix2pix-gp
![here](https://github.com/SerialLain3170/Colorization/blob/master/pix2pix-gp/result.png)

- pix2pixHD

![here](https://github.com/SerialLain3170/Line-to-Color/blob/master/pix2pixHD/visualize_125.png)

## Experiment with atari

### Motivation
Watching results of experiments above, even with using adversarial loss, it seems that neural network falls local minimum. Some degrees of variations in color may exist, neural networks seem to learn to colorize single color to any regions in single character. I find it difficult to train mapping from line art to color image without hint. Therefore, I consider taking the hint, atari, as input of neural network.

### Methods
- [x] UserHint

### Results
- UserHint
![here](https://github.com/SerialLain3170/Line-to-Color/blob/master/UserHint/example.png)

## Experiment with reference

### Motivation
I also consider taking the hint, reference, as input of neural network. First of all, I had tried to implement style2paints V1. However, I had difficulities producing the reproduction of results because training came to collapse. Then, I decide to seek for a substitute for style2paints V1.

### Methods
- [x] style2paints

### Result
- style2paints
![here](https://github.com/SerialLain3170/Colorization/blob/master/style2paints/Result.png)
