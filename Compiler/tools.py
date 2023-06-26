import itertools

class chain(object):
    def __init__(self, *args):
        self.args = args
    def __iter__(self):
        return itertools.chain(*self.args)

class cycle(object):
    def __init__(self, *args):
        self.args = args
    def __iter__(self):
        return itertools.cycle(*self.args)
