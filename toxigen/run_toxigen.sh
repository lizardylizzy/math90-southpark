#!/bin/bash

#SBATCH --cpus-per-task=32                                                                                          
#SBATCH --mem=740G                                                                                                  
#SBATCH --time=23:00:00


# Run your Python script
time python3 discovery_toxigen.py