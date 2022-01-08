from tqdm import tqdm
import torch
import torch.nn as nn
from dataset import get_loader
from model import CifarNet

train_loader, test_loader = get_loader(256)
model = CifarNet(sparsity_ratio=1.0).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(1, 500):
    print(f'Epoch: {epoch}')
    train_loss = 0
    total_num = 0
    correct_num = 0
    total_step = len(train_loader)
    model.train()
    for img_batch, lb_batch in tqdm(train_loader, total=total_step):
        i_batch, lasso = model(img_batch)
        loss = criterion(i_batch, lb_batch) + lasso * (1e-8)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, i_lb_batch = i_batch.max(dim=1)
        total_num += lb_batch.shape[0]
        correct_num += i_lb_batch.eq(lb_batch).sum().item()
    train_loss = train_loss / total_step
    train_acc = 100. * correct_num / total_num
    print('Train acc:%d' %train_acc)

    with torch.no_grad():
        test_loss = 0
        total_num = 0
        correct_num = 0
        total_step = len(test_loader)
        model.eval()
        for img_batch, lb_batch in tqdm(test_loader, total=len(test_loader)):
            i_batch, lasso = model(img_batch, True)
            loss = criterion(i_batch, lb_batch) + lasso * (1e-8)
            test_loss += loss.item()
            _, i_lb_batch = i_batch.max(dim=1)
            total_num += lb_batch.shape[0]
            correct_num += i_lb_batch.eq(lb_batch).sum().item()
        test_loss = test_loss / total_step
        test_acc = 100. * correct_num / total_num
        print('Test acc:%d' % test_acc)