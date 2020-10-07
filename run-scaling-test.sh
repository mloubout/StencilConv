#!/bin/bash -l

set -e

python scaling-devito.py

python scaling-torch.py
