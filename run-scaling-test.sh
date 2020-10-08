#!/bin/bash -l

set -e

python scaling-devito.py

python scaling-torch.py

rclone copy -x -v scaling-torch-no-set-thread-num.txt GTDropbox:scaling-aws/
