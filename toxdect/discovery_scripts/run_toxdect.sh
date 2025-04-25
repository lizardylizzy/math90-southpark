#!/bin/bash

#SBATCH --cpus-per-task=32                                                                                          
#SBATCH --mem=740G                                                                                                  
#SBATCH --time=23:00:00

time python3 final_discovery_toxdect.py
