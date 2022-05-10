import numpy as np
import torch  
import torch.nn as nn   
import torch.utils.data as data_utils  
import torchvision.models as models
import gc
import torchvision.datasets as dataset  
import torchvision.transforms as transforms 
from torch.utils.tensorboard import SummaryWriter
from utils import cutout
from utils import mixup
from utils import cutmix


class CIFAR_ResNet(object):
    def __init__(self, 
                 method = "pure",
                 lr = 1e-2, 
                 lr_decay = 0.95,
                 loss_func = nn.CrossEntropyLoss(), 
                 weight_decay = 1e-4, 
                 epochs = 10, 
                 denote_path = './train_history/pureNet',
                 verbose = True):
        self.method = method
        self.lr = lr
        self.lr_decay = lr_decay
        self.loss_func = loss_func
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.denote_path = denote_path
        self.verbose = verbose
        self.net = models.resnet18(num_classes=100)
        self.val_accuracy_list = []
        transform = transforms.Compose(
            [transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.test_data = dataset.CIFAR100(root="./datasets/cifar100",
                                    train=False,
                                    transform=transform,
                                    download=True)
        if self.method == "cutout":
            transform.transforms.append(cutout.Cutout(8))
        self.train_data = dataset.CIFAR100(root="./datasets/cifar100",
                                    train=True,
                                    transform=transform,
                                    download=True)

        self.batch_size = 32
        self.train_loader = data_utils.DataLoader(dataset=self.train_data, 
                                                batch_size=self.batch_size, 
                                                shuffle=True) 
        self.test_loader = data_utils.DataLoader(dataset=self.test_data, 
                                                batch_size=self.batch_size, 
                                                shuffle=True) 

    def train(self):
        optimizer = torch.optim.SGD(self.net.parameters(), lr = self.lr, momentum = 0.9, weight_decay = self.weight_decay)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        self.net.to(device)
        loss_func = self.loss_func.to(device)

        writer = SummaryWriter(self.denote_path)
        for epoch in range(self.epochs):
            self.net.train()
            for j, (X_batch, y_batch) in enumerate(self.train_loader):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                if self.method == "pure" or self.method == "cutout":
                    optimizer.zero_grad()
                    y_batch_pred = self.net(X_batch)
                    train_loss = self.loss_func(y_batch_pred, y_batch)
                if self.method == "mixup":
                    X_batch, y_a, y_b, lam = mixup.mixup_data(X_batch, y_batch)
                    optimizer.zero_grad()
                    y_batch_pred = self.net(X_batch)
                    train_loss = mixup.mixup_loss(loss_func, y_batch_pred, y_a, y_b, lam)
                if self.method == "cutmix":
                    idx = np.random.rand()
                    if idx < 0.5:
                        X_batch, y_a, y_b, lam = cutmix.cutmix_data(X_batch, y_batch)
                        optimizer.zero_grad()
                        y_batch_pred = self.net(X_batch)
                        train_loss = cutmix.cutmix_loss(loss_func, y_batch_pred, y_a, y_b, lam)
                    else:
                        optimizer.zero_grad()
                        y_batch_pred = self.net(X_batch)
                        train_loss = self.loss_func(y_batch_pred, y_batch)
                train_loss.backward()
                optimizer.step()

                _, predicted = torch.max(y_batch_pred.data, dim=1)
                batch_acc = 100 * (  (predicted == y_batch).sum().item() / y_batch.size()[0] )

                writer.add_scalar('train_loss', train_loss.item(), global_step=int(epoch * len(self.train_data) / self.batch_size) + j)
                writer.add_scalar('train_acc', batch_acc, global_step=int(epoch * len(self.train_data) / self.batch_size) + j)

                if self.verbose and (j % 500) == 0:
                    # print("In epoch %d batch %d , loss = %f, acc = %f" %(epoch, j, train_loss.item(), batch_acc))
                    print("In epoch %d batch %d, batch loss = %.4f, batch acc = %.2f %%" %(epoch+1, j, train_loss.item(), batch_acc))
            
            self.net.eval()
            # 进行评测的时候网络不更新梯度
            with torch.no_grad():
                corr_count = 0
                for j, test in enumerate(self.test_loader):
                    X_test, y_test = test
                    X_test = X_test.to(device)
                    y_test = y_test.to(device)
                    outputs = self.net(X_test)
                    _, y_pred = torch.max(outputs.data, 1)
                    corr_count += (y_pred == y_test).sum().item()
                    test_loss = self.loss_func(outputs, y_test)
                    test_batch_acc = 100 * (  (y_pred == y_test).sum().item() / y_test.size()[0] )
                    writer.add_scalar('test_loss', test_loss.item(), global_step=int(epoch * len(self.test_data) / self.batch_size) + j)
                    writer.add_scalar('test_acc', test_batch_acc, global_step=int(epoch * len(self.test_data) / self.batch_size) + j)
                val_acc = 100 * (corr_count  / len(self.test_data))
                self.val_accuracy_list.append(val_acc)
                print(
                    "(Epoch %d / %d) val_acc: %.2f %%"
                    % (epoch+1, self.epochs, val_acc)
                )
            for p in optimizer.param_groups:
                p['lr'] *= self.lr_decay
            gc.collect()
            torch.cuda.empty_cache()
        writer.close()

    def save(self, name):
        torch.save(self.net.state_dict(), "%s.pkl"%(name))
    
    def load(path):
        net = models.resnet18(num_classes=100)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.load_state_dict(torch.load(path, map_location=device))
        return net