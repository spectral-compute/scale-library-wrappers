# Maths wrappers tests

These are split per library so that bugs in the initialization of one ROCm library doesn't cause every test to fail in a
hard-to-debug way. This includes implicit initialization that happens the first time a function runs.


## Maths wrappers common files

Tests for the common parts of the implementation, for example the HIP/CUDA stream glue.


## Maths wrappers test programs

Each library has a test program so that failure to initialize one does not break every test.
