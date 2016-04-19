import cPickle as pkl
import gzip


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self, source, target,
                 dict,
                 batch_size=128,
                 n_words=-1):
        self.source = fopen(source, 'r')
        self.target = fopen(target, 'r')
        with open(dict, 'rb') as f:
            self.dict = pkl.load(f)
        self.batch_size = batch_size
        self.n_words = n_words
        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
        self.target.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                ss = self.source.readline()
                if ss == "":
                    raise IOError
                ss = ss.strip().split()
                ss = [self.dict[w] if w in self.dict else 1
                      for w in ss]
                if self.n_words > 0:
                    ss = [w if w < self.n_words else 1 for w in ss]

                # read from source file and map to word index
                tt = self.target.readline()
                if tt == "":
                    raise IOError
                tt = tt.strip().split()
                tt = [self.dict[w] if w in self.dict else 1
                      for w in tt]
                if self.n_words > 0:
                    tt = [w if w < self.n_words else 1 for w in tt]

                source.append(ss)
                target.append(tt)

                if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(source) <= 0 or len(target) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source, target
