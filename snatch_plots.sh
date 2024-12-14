#!/bin/bash

sshpass -p "Sulliv@nTr1smegestus" scp -o StrictHostKeyChecking=no sgagnonhartman@trantor01.sns.it:~/galaxy_bias/*.npy ./
python quick_laelf_and_ewpdf.py
