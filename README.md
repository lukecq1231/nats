# Distraction-Based Neural Networks for Modeling Documents

Source code for "Distraction-Based Neural Networks for Modeling Documents" runnable on GPU and CPU. 
If you use this code as part of any published research, please acknowledge the following paper.

**"Distraction-Based Neural Networks for Modeling Documents"**  
Qian Chen, Xiaodan Zhu, Zhenhua Ling, Si Wei, Hui Jiang. *IJCAI (2016)*

    @InProceedings{Chen-Qian:2016:IJCAI,
      author    = {Chen, Qian and Zhu, Xiaodan and Ling, Zhenhua and Wei, Si and Jiang, Hui},
      title     = {Distraction-Based Neural Networks for Modeling Documents},
      booktitle = {Proceedings of the 25th International Joint Conference on Artificial Intelligence (IJCAI 2015)},
      month     = {July},
      year      = {2016},
      address   = {New York, NY},
      publisher = {AAAI}
    }
    
Homepage of the Qian Chen, http://home.ustc.edu.cn/~cq1231/

## Dependencies

This code is written in python. To use it you will need:

- Python 2.6/2.7
- [NumPy](http://www.numpy.org/)
- [scipy](http://www.scipy.org/)
- [argparse](https://www.google.ca/search?q=argparse&oq=argparse&aqs=chrome..69i57.1260j0j1&sourceid=chrome&es_sm=122&ie=UTF-8#q=argparse+pip)
- [Theano](http://www.deeplearning.net/software/theano/)

## Running the Script
### Build dictionary
```
cd data
python build_dictionary.py toy_train_input.txt
```
### Train model
Some important path is needed to set in *train_nats.py*.
- *datasets*: training file of input and output
- *valid_datasets*: validation file of input and output
- *dictionary*: dictionary file
- *model*: saved model

If you don't have [cuDNN](https://developer.nvidia.com/cudnn), please comment the cuDNN configuation in *train.sh*.
```
cd scripts
bash train.sh
```
### Test model
Some variable is needed to set in *test.sh*.
- *KL*:  $\lambda_1$, the parameter of Kullback-Leibler (KL) divergence of attention weight vector
- *CTX*: $\lambda_2$, the parameter of Cosine distance of content vector
- *STATE*: $\lambda_3$, the parameter of Cosine distance of hidden state vector
- *ROOT*: root directory of directory
- *MODEL*: saved model
- *DIC*: dictionary file
- *INPUT*: test file of input
- *TEMP*: intermediate file of generated summary in testing set
- *GEN*: final file of generated summary in testing set
- *REF*: test file of reference summary
```
cd scripts
bash test.sh
```
### Actual Corpus Download
- [LCSTS](http://icrc.hitsz.edu.cn/Article/show/139.html): A Large-Scale Chinese Short Text Summarization Dataset
- [CNN/DailyMail](https://github.com/deepmind/rc-data/): This repository contains a script to download CNN and Daily Mail articles from the Wayback Machine.

