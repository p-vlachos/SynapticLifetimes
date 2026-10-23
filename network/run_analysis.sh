#!/bin/bash

# Run from inside a simulation directory (completed/<timestamp>/),
# uses the analysis code that was copied with the simulation (src/)

PYTHONPATH="$PWD/src" python -m analysis.t5_peaks
