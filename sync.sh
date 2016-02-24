#!/bin/bash

rsync -avz --include='result.*' --include='benchmark' --include='/*' --exclude='*' sabauman@karst.uits.iu.edu:/N/u/sabauman/Karst/src/gradual-typing-performance/benchmarks/ ./results
