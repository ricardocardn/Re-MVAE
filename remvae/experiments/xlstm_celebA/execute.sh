#!/bin/bash
PYTHONPATH=../.. python train.py args.json && \
PYTHONPATH=../.. python visualize.py args.json && \
PYTHONPATH=../.. python eval.py args.json