import numpy as np
import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class PyToFFLM(torch.nn.Module):

    def __init__(self, input_size, output_size, memory=2):
        super().__init__()
        self.em = torch.nn.Linear(input_size, output_size)
        self.h = torch.nn.Linear(memory*output_size, output_size)
        self.a1 = torch.nn.ReLU()
        self.out = torch.nn.Linear(output_size, input_size)
        self.sm = torch.nn.Softmax()
        self.memory = memory

    def forward(self, x):
        e = torch.cat([self.em(x[:, i]) for i in range(self.memory)], 1)
        z1 = self.h(e)
        a1 = self.a1(z1)
        z2 = self.out(a1)
        out = self.sm(z2)
        return out

    def test_mode(self):
        self.em.bias.data = torch.ones(self.em.bias.shape)
        self.em.weight.data = torch.ones(self.em.weight.shape)
        self.h.bias.data = torch.ones(self.h.bias.shape)
        self.h.weight.data = torch.ones(self.h.weight.shape)
        self.out.bias.data = torch.ones(self.out.bias.shape)
        self.out.weight.data = torch.ones(self.out.weight.shape)


def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    """
    Train our model
    """
    model.train()
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_train += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            print('{}  |  Epoch {}  |  Training loss {:.3f}'.format(
                0, epoch,
                loss_train / len(train_loader)))


m1 = PyToFFLM(6, 2, 3)

x = torch.tensor(np.array([[[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1]],
              [[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]],
              [[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]],
              [[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]]])).type(torch.FloatTensor)
m1.test_mode()
out = m1(x)
print(out)
