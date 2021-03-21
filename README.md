## Lets learn to use distributed Pytorch

The University of Sheffield has a high spec computer lab (Computers with Nvidia P4000 GPU's) so I want to use them to train deep learning models faster. Each computer has one GPU but at night no one is there so multiple can be used so lets try to make a small cluster.

For a really good walk through of Distributed data parallel (DDP) training in Pytorch go check out [yangkky](https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html) on GitHub his tutorial is good. That is what I have used here.

I can remote into the UNI computers from home so I can test easily.  To run install [Pytorch](https://pytorch.org/get-started/locally/) (with CUDA)  and TQDM (`pip install tqdm`), then run the train.py script on each node (does not work on windows at this point in time :weary:) .

Run using:

`python train.py -n <number_of_nodes> -g <Number_of_GPUS_per_node> -nr <node_rank> --epochs`

TODO:

- [x] Make basic training script for MNIST/FASION MNIST. (Don't really care if the network is good just that it is trainable)
- [x] Convert training script to use multiple GPU's/nodes
- [x] Test locally (Ubuntu)
- [x] Test on UNI Windows computers.
- [ ] Go into UNI and boot into Ubuntu see how many nodes I can add (cant do remotely)
- [ ] Test speed increase with varying node numbers

Currently DDP does not work on Windows so i'll have to go in to try this on the computers ....... to be continued.  

