from tqdm import tqdm
import os
import argparse
import torch
import torch.nn as nn
import torchvision
import torch.multiprocessing as mp
import torch.distributed as dist
import torchvision.transforms as transforms

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv= nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
        )
        self.fc = nn.Sequential(
                nn.Linear(28*28*32, 10))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def train(gpu, args):
    print('Starting process')
    rank = args.nrank * args.gpus + gpu
    print(rank)
    dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=args.world_size,
            rank=rank
    )

    divice = 'cuda'
    bs = 1024

    m = Model().to(divice)
    print('made model')

    criterion = nn.CrossEntropyLoss().to(divice)
    optimizer = torch.optim.SGD(m.parameters(), lr=0.001)

    # model parallel
    m = nn.parallel.DistributedDataParallel(m, device_ids=[gpu])
    print('made p model')

    train_dataset = torchvision.datasets.FashionMNIST(root='./data',
                                                      train=True,
                                                      transform=transforms.ToTensor(),
                                                      download=True)

    print('made dataset')
    train_sampler = torch.utils.data.DistributedSampler(dataset=train_dataset,
                                                        num_replicas=args.world_size,

                                                        rank=rank)
    print('made smapler')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=bs,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)
    print('loader')
    for epoch in range(args.epochs):
        print(f'Training epoch {epoch}')
        t_loss = 0
        for im, labels in tqdm(train_loader):
            im = im.to(divice)
            labels = labels.to(divice)

            outputs = m(im)
            loss = criterion(outputs, labels)
            t_loss = (t_loss + loss)/2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Loss {t_loss}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int)
    parser.add_argument('-g', '--gpus', default=1, type=int)
    parser.add_argument('-nr', '--nrank', default=0, type=int)
    parser.add_argument('--epochs', default=10, type=int)

    args = parser.parse_args()
    
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '192.168.0.8'
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(train, nprocs=args.gpus, args=(args,))



