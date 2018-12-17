# ANN : Pre-trained VGG-19 (http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat)

Main Jupyter Notebbok takes care of downloading the VGG weights.


# Main Jupyter Notebook : results_demo.ipynb

#### Code Documentation ######

hyperparameters.py : This file has all the hyperparemeters used in the project.
input_processor.py : Takes care of reading and processing inputs.
loss_functions_and_optimizers.py : defines the content and style loss. Also defines the Adam optimizer.
transfer_learning_utils.py : loads the weights of VGG-19 trained on imagenet dataset.
style_transfer.py : Main artistic style transfer engine. Has the main top level function and also the optimization loop.


NOTE : The IPython kernel may die sometimes due to heavy load; please restart kernel and re-run the block of code.
