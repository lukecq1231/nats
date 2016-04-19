import numpy
import os

from nats import train

def main(job_id, params):
    print params
    validerr = train(saveto=params['model'][0],
                                        reload_=params['reload'][0],
                                        dim_word=params['dim_word'][0],
                                        dim=params['dim'][0],
                                        dim_att=params['dim_att'][0],
                                        patience=params['patience'][0],
                                        n_words=params['n-words'][0],
                                        decay_c=params['decay-c'][0],
                                        clip_c=params['clip-c'][0],
                                        lrate=params['learning-rate'][0],
                                        optimizer=params['optimizer'][0], 
                                        maxlen=500,
                                        batch_size=20,
                                        valid_batch_size=20,
                    datasets=['/disk1/%s/nats/data/toy_train_input.txt'%os.environ['USER'], 
                    '/disk1/%s/nats/data/toy_train_output.txt'%os.environ['USER']],
                    valid_datasets=['/disk1/%s/nats/data/toy_validation_input.txt'%os.environ['USER'], 
                    '/disk1/%s/nats/data/toy_validation_output.txt'%os.environ['USER']],
                    dictionary='/disk1/%s/nats/data/toy_train_input.txt.pkl'%os.environ['USER'], 
                                        validFreq=10,
                                        dispFreq=1,
                                        saveFreq=10,
                                        sampleFreq=10,
                                        use_dropout=params['use-dropout'][0])
    return validerr

if __name__ == '__main__':
    main(0, {
        'model': ['/disk1/%s/myopensource/nats/models/model.npz'%os.environ['USER']],
        'dim_word': [120],
        'dim': [600],
        'dim_att': [100],
        'n-words': [25000], 
        'patience': [1],
        'optimizer': ['adadelta'],
        'decay-c': [0.], 
        'clip-c': [100.], 
        'use-dropout': [False],
        'learning-rate': [0.0001],
        'reload': [False],
        })


