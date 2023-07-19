# Transformer Model

## Purpose
The Transformer model has been trained on the book 'Alice in Wonderland'. Given any context, it will iteratively generate new tokens that will mimic the writing style and vocabulary found in the book. The current model runs with a training error of ~0.14.

<img src="https://tse4.mm.bing.net/th/id/OIP.wSRJACD0uJih7XnzTrlLkQHaI8?pid=ImgDet&rs=1" width="300" hspace="35"/> <img src="https://i.pinimg.com/236x/33/ed/8a/33ed8a1e450e82a33b66cd729e54af2f--alice-in-wonderland--alice-in-wonderland-wallpaper.jpg" width="255"/> 

## What we have done
* Created a decoder only transformer network based on the paper ['Attention is all you need'](https://arxiv.org/pdf/1706.03762.pdf)
* Created pipelines to ingest the book Alice in Wonderland
* Tokenised at word level (Naive tokenizer)
* Enabled single GPU training (Nvidia Titan XP 12 GB)
* Implemented early stopping
* Conducted hyperparameter optimization

## Key Takeaways
1. Training transformers on language data requires GPU compute power
2. Data pre-processing pipelines must be efficient to scale
3. Distributed GPU training is required for data to scale
4. A transformer is just a fancy autocomplete tool

## Dependencies
`pip install torch numpy collections`

* [pytorch](https://pytorch.org/get-started/locally/) - building and training the transformer model
* [numpy](https://numpy.org/install/) - for numerical computations 
* `collections` - used as part of the tokenising method 

As the text file and model path were saved in Google drive, these dependencies were also required to retrieve the files:
* `os`: Operating system related functions and file operations
* `drive`: Mounting Google Drive in Colab to access files
* `sys`: Accessing system-specific parameters and functions

**Please Note:** A GPU is required to replicate and train this model. 

---

### Example Outputs:

***Prompt:*** *"She is a Queen"* 
  
**Response:** She is a Queen never was so small as this before, never! And I declare it's too bad, that it is!" As she said these words her foot slipped, and in another moment, splash!

***Prompt:*** *"The cat was sad!*
  
**Response:** he cat was sad and round it: there was a king,’ said Alice. ‘I’ve read that in some book, but I don’t remember where.’ ‘Well, it must be removed,’ said the King very decidedly,

***Prompt:*** *"tired"*
  
**Response:** tired of the other: the only difficulty was, that she had not the smallest idea how to set about it; and while she was peering about anxiously among the trees,

---

#### Authors
Arzu Khanna & Joe B Prinable 
