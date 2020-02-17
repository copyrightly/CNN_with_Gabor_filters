# CNN_with_Gabor_filters

This is a 4-layer Convolutional Neural Network(CNN) to train MNIST dataset. I modified the original architecture provided by pytorch tutorial: https://github.com/pytorch/examples/tree/master/mnist

Unlike the conventional convolutional filter (a kXk box with k^2 independent parameters), the first layer uses both the real and imaginary parts of Gabor filter as convolutional filters. The advantage is that we can save about 50% parameters.

The second layer uses conventional convolutional filters to extract high-level features of images

The last two layers are fully connected, followed by softmax to make prediction.

You can simply run the code as:
python gabor_both_cos_sin_two_layers_add_noise.py

Moreover you can test the robustness of the model by adding noise to images, where the noise obeys Gaussian distributuion with parameters mean and std. You can run the code as:
python gabor_both_cos_sin_two_layers_add_noise.py --mean 1.0 --std 1.0

More training arguments can be found in the parser of the code.

# References:
https://github.com/pytorch/examples/tree/master/mnist

https://en.wikipedia.org/wiki/Gabor_filter
