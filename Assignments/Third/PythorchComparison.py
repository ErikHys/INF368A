import numpy as np
import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class PyToFFLM(torch.nn.Module):

    def __init__(self, input_size, output_size, memory=2):
        super().__init__()
        self.em = torch.nn.Linear(input_size, output_size, bias=False)
        self.h = torch.nn.Linear(memory*output_size, output_size)
        self.a1 = torch.nn.ReLU()
        self.out = torch.nn.Linear(output_size, input_size)
        self.sm = torch.nn.Softmax(dim=1)
        self.memory = memory

    def forward(self, x):
        e = torch.cat([self.em(x[:, i]) for i in range(self.memory)], 1)
        print(self.em.weight.shape, "Embed shape")
        z1 = self.h(e)
        print(e.shape, 'e shape')
        print(self.h.weight.shape, "Hidden shape")
        print(z1.shape, "z1, e * hidden ish")
        print(self.out.weight.shape, "Output shape")

        a1 = self.a1(z1)
        z2 = self.out(a1)
        print(z2, "out * z1")
        out = self.sm(z2)
        return out

    def test_mode(self):
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


# m1 = PyToFFLM(6, 2, 3)
# opt = torch.optim.SGD(m1.parameters(), lr=0.9999)
# loss = torch.nn.CrossEntropyLoss()
# x = torch.tensor(np.array([[[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1]]])).type(torch.FloatTensor)
# y = torch.tensor(np.array([0])).type(torch.LongTensor)
# m1.test_mode()
# out = m1(x)
# print(out)
# l = loss(out, y)
# print(l)
# l.backward()
# for p in m1.parameters():
#     print(p.grad, p.grad.shape, p.shape)

