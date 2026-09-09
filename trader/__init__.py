"""Execution layer: turns a target portfolio into broker orders, under limits.

Nothing in this package trades on its own. Every entry point defaults to a dry
run, live mode requires an explicit flag plus a typed confirmation, and the
credentials are read from the environment and never stored in the repository.
The operator holds the keys and presses the button; this code only computes
what the orders should be and refuses the ones that breach a limit.
"""
