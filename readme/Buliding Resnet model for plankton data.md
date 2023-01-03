To be honest, this is the first time I build CNN model with pytorch or tensorflow, as my former background mainly inclined to machine learning but not deep learning. After going through some polular model like Alexnet and GoogleNet and VGG, I finally decide to use ResNet model to solve the problem. The reason I choose this model is based on the very famous paper "Deep Residual Learning for Image Recognition", in this small readme, I will go through my study for this small quiz.
### Before build the model
Lots of people say even high school student can do deep learning, during my study of basic termiology for my model I found its true as there's not so many hardcore math to build the model(if you just want to use the model). I go through the difination of tensor (and TPU), basic image processing and the structure of CNN different from ANN I learnt before especially the convolutional layer. I do some math work to understand back propagation and batch normalization, which makes me feel the importance of using tensor.
### Deep Residual Learning for Image Recognition
After reading the paper, I found 3 highlight.
1. very deep (1000) channels.
2. residual block.
3. using batch normalization to speed up.

#### Deep channals
The author first answer my question why can we just simply add more channels 
![Screenshot 2022-12-30 at 19.24.26.png](resources/ECDFBB9C06C7061B9F7D256F309100C0.png =432x296)
From this figure we can see for "plain" network, if we increase the layer from 20 to 56, the training error does not decrease but increase. The author hinted for one reason which is the degradation problem and give solution by the residual structure.
Another reason I think is the unstable gradient during back propagation. The gradient may disappear beacause for the back propagation in different layer, we need to multiple a error gradient less than one, and the gradient may explode if we multiple a error gradient larger than one.

#### Residual Block
![Screenshot 2022-12-30 at 20.34.41.png](resources/6D9A218082FC06EFC9E0863C85FA446B.png =107x96)
##### case of 18 and 34 layer, conv3_x
Input is [56,56,64], output is [28,28,128]
![Screenshot 2022-12-30 at 20.42.08.png](resources/627CBD0B176E05A46392A6328FEB6A5A.png =250x234)
The sidechain(called shortcut in the paper) is the residual chain, with two kinds: the solid line is the pure residual and a dashline is added a 1X1 convolutional kernal, when the channels of two sides of the shortcut are different, making sure the output of the backbone and shortcut have same size
We change stride to change the size of the kernel(stride=2 so change from 56X56 to 28X28)
##### case of 50, 101 and 156 layer, conv_3x 
Input is [56,56,256], output is [28,28,512]
![Screenshot 2022-12-30 at 20.53.21.png](resources/16EF8B09ED5ACBEDE95F252751B9B00E.png =237x241)
So here the difference is we add a 1X1 kernel to match the channel in the backbone in the beginning, from 256 to 128, and another in the end of the backbone to amplify the channel to 512 for next layer.
One thing need tohighlight is the structure is slightly different from the paper, in paper in the backbone, the stride is 211, and we use 121, we use the pytorch webside suggestion, which can enhance around 0.5 accurency in imagenet on top 1.
![Screenshot 2022-12-30 at 23.28.28.png](resources/F2D813B72DF33CB1CCEA6736588590A7.png =598x220)

#### Batch Nomalization
According to my understanding, the aim of batch normalization is make sure our feature map(batch)have mean 0 and variance 1. It is published by Google at 2015. It will accelerate the convergence and accurency of our CNN.
![Screenshot 2022-12-30 at 23.34.23.png](resources/D335811F1B79960575870922694799E6.png =234x185)
We have mean and variance from forward propagation and scale and shift from backword propagation.
Why we need BN?
![Screenshot 2022-12-30 at 23.37.51.png](resources/4158474A9B673384F89DA0DD2BEBF18E.png =237x127)
In a lot of case, we need preprocessing to adjust the data into a typical distribution to acceerate our training. But our feature map will not meet the distribution after the conv1, so BN will adjust the feature map in every layer to make sure it meets the distribution we set before. Maybe mean=0 and variance=1 is not perfect, so after training, we have the scale and shift to fix them.
![Screenshot 2022-12-31 at 00.31.42.png](resources/70EF80CEB0E983BA85ED386D634E0BD1.png =731x391)
### First CNN model training
Based on the model from the paper, I build a CNN model and run the trainning set(I split the data with 80% trainning and 20% testing) and run for 10 epoches. The accurency in last epoch is only 56.78%, which is not good at all, but this is jut a first test, I think I need to make progress in these ways:
1. Preprocessing of the data, some of the attributes we only have 9 figures. Do some rotation, zooming?
2. We can use tranfer learing and load the pretrain weights given by pytorch ResNet34, with our fully connected layer(we have 121 attributes so 121 layers).
3. We can increase the number of epoch to 30 as the accurency doesn't converge yet, which means I need a cloud service, my poor Mac...

#### Data Preprocessing
As lots of our attributes have insufficient images, so we need to enrich our dataset. In the begging of the code, we crop, horizontal flip and normalized(according to the ResNet34 tutorial) the trainning and testing data. 
#### Transfer Learning
In the Pytorch webset we can find the pretrain weights for the ResNet34 network, i download it and apply to the model.
#### Training Epoch
I rent a AWS EC2 cloud server and connect it use SSH and upload my code on it, run it with VS code, this is the most time consuming part for this quiz as I test lots of different servers and building pytorch environment.
### Final results
Accurency 75% with epoch 30.