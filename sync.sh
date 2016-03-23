#!/bin/bash

rsync -avz --include='result*.m2' --include='benchmark' --include='/*' --exclude='*' sabauman@karst.uits.iu.edu:/N/u/sabauman/Karst/src/gradual-typing-performance/benchmarks/ ./results
rsync -avz --include='result*.m2' --include='benchmark' --include='/*' --exclude='*' sabauman@karst.uits.iu.edu:/N/u/sabauman/Karst/src/gradual-typing-performance-v6.2.1/benchmarks/ ./results-6.2.1
