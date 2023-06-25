from Compiler import types, instructions

class Program(object):
    def __init__(self, progname):
        types.program = self
        instructions.program = self
        self.curr_tape = None
        exec(compile(open(progname).read(), progname, 'exec'))
    def malloc(self, *args):
        pass
