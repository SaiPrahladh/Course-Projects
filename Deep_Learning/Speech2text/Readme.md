## Attention-based End-to-End Speech-to-Text Deep Neural Network

Implementation of the research paper 'Listen Attend and Spell'
Ref: https://arxiv.org/abs/1508.01211


The idea is to construct an end to end speech detection using bidirectional lstms as the cell block, and build an encoder-decoder network to construct the final output with an attention mechanism. This is an extensive and a long implementation and has a lot of moving parts. The dataset provided to us consists of the train, validation and test data in numpy format and we have the transcripts for the training and validation data.  We explore a character level implementation here. To deal with the data labels present in string format, we need to convert them in some sort of indexed form while training. So we make use of the ‘transform_letter_to_index’ and ‘create_dictionaries’ helper functions. While performing inference, to visualize in a heuristic sense, if our predictions make sense and have some form of semblance to English, we use the ‘transform_index_to_letter’ helper function.

Dataloader: This is an override of the Dataset class of pytorch. This class has the standard __init__, __len__, and __get_item__ functions. We also define collate functions for the train and test data. We perform the ‘pad_sequence’ operation on the speech data and the corresponding text data (wherever applicable). We return the padded speech, the length of speech data, the padded text and its length. We skip the returns related to the text for the test data.

Hyperparameter: 
Locked_dropout = 0.2
Teacher_forcing(variable): reducing from 90 to 80 to 70 along epochs.
Batch_size = 64
Initial learning rate = 0.001
Criterion = CrossEntropyLoss
Scheduler = ReduceLROnPlateau, reduction factor = 0.75, patience = 1 epoch
Num_epochs = 60

Model: There are 2 blocks which are included within the Seq2Seq class,the Encoder and the Decoder. The attention mechanism is implemented as a part of the decoder block. The model has input dimensions 40 and hidden size = 256. We also have 3 pyramidal BiLSTM blocks invoked in the Encoder block. The implementation of teacher forcing has a variable rate with respect to the number of epochs. There are locked dropout layers invoked in between each of the pBLSTM layer in the Encoder. Implementing weight tying in the decoder block is a beneficial step.

The above configuration achieves a Levenshtein score of 24.3

Visualizing the attention plot gives the following result:
![alt text](https://github.com/SaiPrahladh/Course-Projects/blob/master/Deep_Learning/Speech2text/Attention.png)
