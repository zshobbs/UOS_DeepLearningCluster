from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
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
            

if __name__ == '__main__':
    divice = 'cuda'
    bs = 1024
    epochs = 10

    train_dataset = torchvision.datasets.FashionMNIST(root='./data',
                                                      train=True,
                                                      transform=transforms.ToTensor(),
                                                      download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=bs,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True)

    m = Model().to(divice)

    criterion = nn.CrossEntropyLoss().to(divice)
    optimizer = torch.optim.SGD(m.parameters(), lr=0.001)

    for epoch in range(epochs):
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
