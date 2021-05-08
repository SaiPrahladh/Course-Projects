## Face verification:

#### Problem statement: 
The problem is to verify if two given images are of the same person or
not. This problem can be split into two.
1. The classification problem, where we will train our network to correctly match the image
   to its corresponding label.
2. The verification problem, where we will verify the cosine similarity between the
   embeddings of the images passed through the model. These embeddings are the
   features extracted from the penultimate layer of the model.

The choice of model architecture here is ResNet18.

#### Why ResNet?: 
ResNets have been known to outperform classical CNNs by allowing us to build deeper networks. 
The primary shortcomings of classical CNN network like for example VGG-19 network exhibit 
higher training and validation error as compared to a shallower network. 
The paper on ResNet https://arxiv.org/pdf/1512.03385.pdf says that 
"not all systems are similarly easy to optimize.". This could be because of the problem of
vanishing gradients, and this is where ResNets have an advantage. ResNets enable the gradients
to reach every layer and thus allows us to build deeper networks with better performance.
These articles, along with the above research paper give a much deeper insight into the workings
of ResNets:
1. https://analyticsindiamag.com/why-resnets-are-a-major-breakthrough-in-image-processing/
2. https://towardsdatascience.com/introduction-to-resnets-c0a830a288a4

<img src="https://www.researchgate.net/figure/ResNet-18-Architecture_tbl1_322476121" width="300" height="300" />


The data we are dealing with here is of the dimension 64x64x3

#### Model Description
The ResNet 18 model consists of two basic blocks per layer and has 4
such layers. The basic blocks consist of two convolution operations, each with a kernel size of
3x3 and a padding of 1. The stride is different for both the operations. The first convolution takes
the stride assumed by the input, whereas the second convolution has a stride length of 1. There
is also a shortcut or residue, which is added at each skip connection. This residue is either the
input at the start of the layer or is a result of a convolution operation with kernel size 1. The
residue is the input when there is no dimension change or when the stride is equal to 1.

#### Data Loader:
This implementation uses two dataloaders, one for each of the above mentioned
problems.
For the classification problem, I have used the inbuilt dataset class (datasets.ImageFolder)
provided by PyTorch. Within this dataset class, the transforms applied were as follows.
1. Resizing the image to 224x224x3 in order to allow the usage of a large kernel of size
   7x7 within the architecture.
2. Horizontal Flipping of the image with a probability of 0.3 to ensure robustness in
   classification.
3. Tensor transformation to convert images into tensors.
For the verification problem, I wrote my own dataset class which functions as follows.
1. The user passes the text containing the path of the images to be compared. The dataset
   class first checks if it is a part of the validation dataset or the test dataset.
2. If it is a part of validation dataset, it iterates through the text file line by line and appends
   them into two lists, each corresponding to one image. The labels are appended to a
   separate list of labels.
3. The same process is followed for the test dataset also, with the absence of the labels
   field.
4. Within the get_item method, we first open the image using the Image.open command
   and then return the transformed image (Resizing and conversion to tensor).

#### Hyperparameters:
Initial Learning rate = 0.15
Number of epochs = 15
Loss- Cross Entropy Loss
Optimizer- SGD with Momentum as 0.9
Scheduler- StepLR with step size 1 and gamma (Multiplicative factor of learning rate decay) as 0.85

#### Roc-Auc calculation:
Loop through the dataloader of validation data. Pass the data through
the model after dropping its final layer (linear layer) to acquire the embeddings. Once we get all
the embeddings, use the nn.CosineSimilarity function to calculate the similarity metric and
append this to a list. Once we have the similarity scores for all the image pairs, we can flatten
the list and use the roc_auc_score from sklearn and pass the similarities list and the labels to
calculate the score.
For the test data, just get the cosine similarities list as the final result.
