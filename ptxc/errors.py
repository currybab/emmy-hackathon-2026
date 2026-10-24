class CompileError(Exception):
    """The input cannot be compiled; the message says what and where.

    The compiler raises this instead of emitting PTX it is unsure of: a loud failure on an
    unsupported construct beats a kernel that assembles and silently computes the wrong thing.
    """
