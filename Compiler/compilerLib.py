import inspect
import os
import re
import sys
import tempfile
from optparse import OptionParser

from Compiler.exceptions import CompilerError

from .GC import types as GC_types
from .program import Program, defaults


class Compiler:
    def __init__(self, custom_args=None, usage=None):
        if usage:
            self.usage = usage
        else:
            self.usage = "usage: %prog [options] filename [args]"
        self.custom_args = custom_args
        self.build_option_parser()
        self.VARS = {}

    def build_option_parser(self):
        parser = OptionParser(usage=self.usage)
        parser.add_option(
            "-n",
            "--nomerge",
            action="store_false",
            dest="merge_opens",
            default=defaults.merge_opens,
            help="don't attempt to merge open instructions",
        )
        parser.add_option("-o", "--output", dest="outfile", help="specify output file")
        parser.add_option(
            "-a",
            "--asm-output",
            dest="asmoutfile",
            help="asm output file for debugging",
        )
        parser.add_option(
            "-g",
            "--galoissize",
            dest="galois",
            default=defaults.galois,
            help="bit length of Galois field",
        )
        parser.add_option(
            "-d",
            "--debug",
            action="store_true",
            dest="debug",
            help="keep track of trace for debugging",
        )
        parser.add_option(
            "-c",
            "--comparison",
            dest="comparison",
            default="log",
            help="comparison variant: log|plain|inv|sinv",
        )
        parser.add_option(
            "-M",
            "--preserve-mem-order",
            action="store_true",
            dest="preserve_mem_order",
            default=defaults.preserve_mem_order,
            help="preserve order of memory instructions; possible efficiency loss",
        )
        parser.add_option(
            "-O",
            "--optimize-hard",
            action="store_true",
            dest="optimize_hard",
            help="currently not in use",
        )
        parser.add_option(
            "-u",
            "--noreallocate",
            action="store_true",
            dest="noreallocate",
            default=defaults.noreallocate,
            help="don't reallocate",
        )
        parser.add_option(
            "-m",
            "--max-parallel-open",
            dest="max_parallel_open",
            default=defaults.max_parallel_open,
            help="restrict number of parallel opens",
        )
        parser.add_option(
            "-D",
            "--dead-code-elimination",
            action="store_true",
            dest="dead_code_elimination",
            default=defaults.dead_code_elimination,
            help="eliminate instructions with unused result",
        )
        parser.add_option(
            "-p",
            "--profile",
            action="store_true",
            dest="profile",
            help="profile compilation",
        )
        parser.add_option(
            "-s",
            "--stop",
            action="store_true",
            dest="stop",
            help="stop on register errors",
        )
        parser.add_option(
            "-R",
            "--ring",
            dest="ring",
            default=defaults.ring,
            help="bit length of ring (default: 0 for field)",
        )
        parser.add_option(
            "-B",
            "--binary",
            dest="binary",
            default=defaults.binary,
            help="bit length of sint in binary circuit (default: 0 for arithmetic)",
        )
        parser.add_option(
            "-G",
            "--garbled-circuit",
            dest="garbled",
            action="store_true",
            help="compile for binary circuits only (default: false)",
        )
        parser.add_option(
            "-F",
            "--field",
            dest="field",
            default=defaults.field,
            help="bit length of sint modulo prime (default: 64)",
        )
        parser.add_option(
            "-P",
            "--prime",
            dest="prime",
            default=defaults.prime,
            help="prime modulus (default: not specified)",
        )
        parser.add_option(
            "-I",
            "--insecure",
            action="store_true",
            dest="insecure",
            help="activate insecure functionality for benchmarking",
        )
        parser.add_option(
            "-b",
            "--budget",
            dest="budget",
            default=defaults.budget,
            help="set budget for optimized loop unrolling " "(default: 100000)",
        )
        parser.add_option(
            "-X",
            "--mixed",
            action="store_true",
            dest="mixed",
            help="mixing arithmetic and binary computation",
        )
        parser.add_option(
            "-Y",
            "--edabit",
            action="store_true",
            dest="edabit",
            help="mixing arithmetic and binary computation using edaBits",
        )
        parser.add_option(
            "-Z",
            "--split",
            default=defaults.split,
            dest="split",
            help="mixing arithmetic and binary computation "
            "using direct conversion if supported "
            "(number of parties as argument)",
        )
        parser.add_option(
            "--invperm",
            action="store_true",
            dest="invperm",
            help="speedup inverse permutation (only use in two-party, "
            "semi-honest environment)"
        )
        parser.add_option(
            "-C",
            "--CISC",
            action="store_true",
            dest="cisc",
            help="faster CISC compilation mode",
        )
        parser.add_option(
            "-K",
            "--keep-cisc",
            dest="keep_cisc",
            help="don't translate CISC instructions",
        )
        parser.add_option(
            "-l",
            "--flow-optimization",
            action="store_true",
            dest="flow_optimization",
            help="optimize control flow",
        )
        parser.add_option(
            "-v",
            "--verbose",
            action="store_true",
            dest="verbose",
            help="more verbose output",
        )
        self.parser = parser

    def parse_args(self):
        self.options, self.args = self.parser.parse_args(self.custom_args)
        if self.options.optimize_hard:
            print("Note that -O/--optimize-hard currently has no effect")

    def build_program(self, name=None):
        self.prog = Program(self.args, self.options, name=name)

    def build_vars(self):
        from . import comparison, floatingpoint, instructions, library, types

        # add all instructions to the program VARS dictionary
        instr_classes = [
            t[1] for t in inspect.getmembers(instructions, inspect.isclass)
        ]

        for mod in (types, GC_types):
            instr_classes += [
                t[1]
                for t in inspect.getmembers(mod, inspect.isclass)
                if t[1].__module__ == mod.__name__
            ]

        instr_classes += [
            t[1]
            for t in inspect.getmembers(library, inspect.isfunction)
            if t[1].__module__ == library.__name__
        ]

        for op in instr_classes:
            self.VARS[op.__name__] = op

        # backward compatibility for deprecated classes
        self.VARS["sbitint"] = GC_types.sbitintvec
        self.VARS["sbitfix"] = GC_types.sbitfixvec

        # add open and input separately due to name conflict
        self.VARS["vopen"] = instructions.vasm_open
        self.VARS["gopen"] = instructions.gasm_open
        self.VARS["vgopen"] = instructions.vgasm_open
        self.VARS["ginput"] = instructions.gasm_input

        self.VARS["comparison"] = comparison
        self.VARS["floatingpoint"] = floatingpoint

        self.VARS["program"] = self.prog
        if self.options.binary:
            self.VARS["sint"] = GC_types.sbitintvec.get_type(int(self.options.binary))
            self.VARS["sfix"] = GC_types.sbitfixvec
            for i in [
                "cint",
                "cfix",
                "cgf2n",
                "sintbit",
                "sgf2n",
                "sgf2nint",
                "sgf2nuint",
                "sgf2nuint32",
                "sgf2nfloat",
                "cfloat",
                "squant",
            ]:
                del self.VARS[i]

    def prep_compile(self, name=None):
        self.parse_args()
        if len(self.args) < 1 and name is None:
            self.parser.print_help()
            exit(1)
        self.build_program(name=name)
        self.build_vars()

    def compile_file(self):
        """Compile a file and output a Program object.

        If options.merge_opens is set to True, will attempt to merge any
        parallelisable open instructions."""
        print("Compiling file", self.prog.infile)

        with open(self.prog.infile, "r") as f:
            changed = False
            if self.options.flow_optimization:
                output = []
                if_stack = []
                for line in f:
                    if if_stack and not re.match(if_stack[-1][0], line):
                        if_stack.pop()
                    m = re.match(
                        r"(\s*)for +([a-zA-Z_]+) +in " r"+range\(([0-9a-zA-Z_]+)\):",
                        line,
                    )
                    if m:
                        output.append(
                            "%s@for_range_opt(%s)\n" % (m.group(1), m.group(3))
                        )
                        output.append("%sdef _(%s):\n" % (m.group(1), m.group(2)))
                        changed = True
                        continue
                    m = re.match(r"(\s*)if(\W.*):", line)
                    if m:
                        if_stack.append((m.group(1), len(output)))
                        output.append("%s@if_(%s)\n" % (m.group(1), m.group(2)))
                        output.append("%sdef _():\n" % (m.group(1)))
                        changed = True
                        continue
                    m = re.match(r"(\s*)elif\s+", line)
                    if m:
                        raise CompilerError("elif not supported")
                    if if_stack:
                        m = re.match("%selse:" % if_stack[-1][0], line)
                        if m:
                            start = if_stack[-1][1]
                            ws = if_stack[-1][0]
                            output[start] = re.sub(
                                r"^%s@if_\(" % ws, r"%s@if_e(" % ws, output[start]
                            )
                            output.append("%s@else_\n" % ws)
                            output.append("%sdef _():\n" % ws)
                            continue
                    output.append(line)
                if changed:
                    infile = tempfile.NamedTemporaryFile("w+", delete=False)
                    for line in output:
                        infile.write(line)
                    infile.seek(0)
                else:
                    infile = open(self.prog.infile)
            else:
                infile = open(self.prog.infile)

        # make compiler modules directly accessible
        sys.path.insert(0, "Compiler")
        # create the tapes
        exec(compile(infile.read(), infile.name, "exec"), self.VARS)

        if changed and not self.options.debug:
            os.unlink(infile.name)

        return self.finalize_compile()

    def register_function(self, name=None):
        """
        To register a function to be compiled, use this as a decorator.
        Example:

            @compiler.register_function('test-mpc')
            def test_mpc(compiler):
                ...
        """

        def inner(func):
            self.compile_name = name or func.__name__
            self.compile_function = func
            return func

        return inner

    def compile_func(self):
        if not (hasattr(self, "compile_name") and hasattr(self, "compile_func")):
            raise CompilerError(
                "No function to compile. "
                "Did you decorate a function with @register_fuction(name)?"
            )
        self.prep_compile(self.compile_name)
        print(
            "Compiling: {} from {}".format(self.compile_name, self.compile_func.__name__)
        )
        self.compile_function()
        self.finalize_compile()

    def finalize_compile(self):
        self.prog.finalize()

        if self.prog.req_num:
            print("Program requires at most:")
            for x in self.prog.req_num.pretty():
                print(x)

        if self.prog.verbose:
            print("Program requires:", repr(self.prog.req_num))
            print("Cost:", 0 if self.prog.req_num is None else self.prog.req_num.cost())
            print("Memory size:", dict(self.prog.allocated_mem))

        return self.prog
