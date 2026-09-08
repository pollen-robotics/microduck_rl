"""Serving a simulated microduck body to the real `robotd`.

This is the other half of `microduck`'s `robotd --sim`: MuJoCo holds the body, the daemon holds
everything else, and a socket between them replaces the servo bus. See `docs/design/simulation.md`
in the microduck repo for the protocol and for what the twin is and is not.
"""
