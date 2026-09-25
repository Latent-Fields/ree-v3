"""Developmental-stage sources for REE-v3.

Currently hosts:
  - StructuredBabbler (``structured_babbling.py``; coupled-loop-repair campaign W2a):
    a class-balanced, persistent-run motor generator. Constructed by ``REEAgent`` only
    when ``structured_babbling_enabled`` is True; called by nothing in ree_core.

Deliberately imports nothing, so importing the package never loads a source.
"""
