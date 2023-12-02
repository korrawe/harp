# HandNeRF for sequences in HARP
This repository contains the code for my project on hand-object interaction, where I was responsible for building up a NeRF system for hands.

The main code is in optimize_sequence_nerf.py, which load the released sequence from harp, and optimize the NeRF system with MANO hand model.

![HandNeRF](./HandNeRF.mp4)

## Usage
First, download the released sequences from HARP dataset:

## Credits
HARP: https://github.com/korrawe/harp.git for its released sequences and data loading codes.

DS_NeRF: https://github.com/dunbar12138/DSNeRF.git for its compact NeRF implementation.

NeuMan: https://github.com/apple/ml-neuman.git for its inspiring work on human modeling and partial codes for lbs.