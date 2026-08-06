#!/bin/bash
cd /gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
envs/reasoning3b/bin/python -c "import nltk; nltk.download(['punkt', 'punkt_tab'], quiet=True); print('nltk data OK')"
