'''
Summary a source file using a summarization model.
'''
import argparse

import numpy
import cPickle as pkl

from nats import (build_sampler, gen_sample, load_params,
                 init_params, init_tparams)

from multiprocessing import Process, Queue


def translate_model(queue, rqueue, pid, model, options, k, normalize, kl_factor, ctx_factor, state_factor):

    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    trng = RandomStreams(1234)

    # allocate model parameters
    params = init_params(options)

    # load model parameters and set theano shared variables
    params = load_params(model, params)
    tparams = init_tparams(params)

    # word index
    f_init, f_next = build_sampler(tparams, options, trng)

    def _translate(seq):
        # sample given an input sequence and obtain scores
        sample, score, alphas = gen_sample(tparams, f_init, f_next,
                                   numpy.array(seq).reshape([len(seq), 1]),
                                   options, trng=trng, k=k, maxlen=100,
                                   stochastic=False, argmax=False, use_unk=True, kl_factor=kl_factor, ctx_factor=ctx_factor, state_factor=state_factor)

        # normalize scores according to sequence lengths
        if normalize:
            lengths = numpy.array([len(s) for s in sample])
            score = score / lengths
        sidx = numpy.argmin(score)
        align_pos = []
        for alpha in alphas[sidx]:
            align_pos.append(numpy.argmax(alpha))
        return sample[sidx], align_pos

    while True:
        req = queue.get()
        if req is None:
            break

        idx, x = req[0], req[1]
        print pid, '-', idx
        seq, pos = _translate(x)

        rqueue.put((idx, seq, pos))

    return


def main(model, dictionary, source_file, saveto, k=5,
         normalize=False, n_process=5, chr_level=False, kl_factor=0, ctx_factor=0, state_factor=0):

    # load model model_options
    with open('%s.pkl' % model, 'rb') as f:
        options = pkl.load(f)

    # load dictionary and invert
    with open(dictionary, 'rb') as f:
        word_dict = pkl.load(f)
    word_idict = dict()
    for kk, vv in word_dict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    # create input and output queues for processes
    queue = Queue()
    rqueue = Queue()
    processes = [None] * n_process
    for midx in xrange(n_process):
        processes[midx] = Process(
            target=translate_model,
            args=(queue, rqueue, midx, model, options, k, normalize, kl_factor, ctx_factor, state_factor))
        processes[midx].start()

    # utility function
    def _seqs2words(caps, pos):
        capsw = []
        for cc, pp in zip(caps, pos):
            ww = []
            for w, p in zip(cc, pp):
                if w == 0:
                    break
                ww.append(word_idict[w])
                ww.append('[{0}]'.format(p))
            capsw.append(' '.join(ww))
        return capsw

    def _send_jobs(fname):
        with open(fname, 'r') as f:
            for idx, line in enumerate(f):
                if chr_level:
                    words = list(line.decode('utf-8').strip())
                else:
                    words = line.strip().split()
                x = map(lambda w: word_dict[w] if w in word_dict else 1, words)
                x = map(lambda ii: ii if ii < options['n_words'] else 1, x)
                x += [0]
                queue.put((idx, x))
        return idx+1

    def _finish_processes():
        for midx in xrange(n_process):
            queue.put(None)

    def _retrieve_jobs(n_samples):
        trans = [None] * n_samples
        pos = [None] * n_samples
        for idx in xrange(n_samples):
            resp = rqueue.get()
            trans[resp[0]] = resp[1]
            pos[resp[0]] = resp[2]
            if numpy.mod(idx, 10) == 0:
                print 'Sample ', (idx+1), '/', n_samples, ' Done'
        return trans, pos

    print 'Inferece ', source_file, '...'
    n_samples = _send_jobs(source_file)
    trans, pos = _retrieve_jobs(n_samples)
    trans = _seqs2words(trans, pos)
    _finish_processes()
    with open(saveto, 'w') as f:
        print >>f, '\n'.join(trans)
    print 'Done'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=5)
    parser.add_argument('-p', type=int, default=5)
    parser.add_argument('-l', type=float, default=0)
    parser.add_argument('-x', type=float, default=0)
    parser.add_argument('-s', type=float, default=0)
    parser.add_argument('-n', action="store_true", default=False)
    parser.add_argument('-c', action="store_true", default=False)
    parser.add_argument('model', type=str)
    parser.add_argument('dictionary', type=str)
    parser.add_argument('source', type=str)
    parser.add_argument('saveto', type=str)

    args = parser.parse_args()

    main(args.model, args.dictionary, args.source,
         args.saveto, k=args.k, normalize=args.n, n_process=args.p, 
         chr_level=args.c, kl_factor=args.l, ctx_factor=args.x, state_factor=args.s)
