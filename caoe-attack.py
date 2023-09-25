import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from pdb import set_trace as st


random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
device = torch.device('cuda:0')

""" Models """
class PassNet(nn.Module):
    def __init__(self):
        super(PassNet, self).__init__()
    def forward(self, x):
        return x

class Net(nn.Module):
    def __init__(self, nclass=10, c=1):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(c, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclass)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, c=1):
        super().__init__()
        """ Encoder """
        self.e1 = encoder_block(c, 64)
        self.e2 = encoder_block(64, 128)
        # self.e3 = encoder_block(128, 256)
        # self.e4 = encoder_block(256, 512)
        """ Bottleneck """
        # self.b = conv_block(512, 1024)
        self.b = conv_block(128, 256)
        """ Decoder """
        # self.d1 = decoder_block(1024, 512)
        # self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)
        # """ Classifier """
        self.outputs = nn.Conv2d(64, c, kernel_size=1, padding=0)
    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        # s3, p3 = self.e3(p2)
        # s4, p4 = self.e4(p3)
        """ Bottleneck """
        b = self.b(p2)
        """ Decoder """
        # d1 = self.d1(b, s4)
        # d2 = self.d2(d1, s3)
        d3 = self.d3(b, s2)
        d4 = self.d4(d3, s1)
        """ Classifier """
        outputs = 0.5*F.tanh(self.outputs(d4)) + inputs
        # outputs = torch.clamp(outputs, min=0, max=1)
        # print(f'{torch.max(outputs)}-{torch.min(outputs)}')
        # st()
        # outputs = d4 + inputs
        # print('in',inputs.shape)
        # print('s1',s1.shape)
        # print('p1',p1.shape)
        # print('s2',s2.shape)
        # print('p2',p2.shape)
        # print('br',b.shape)
        # print('d3',d3.shape)
        # print('d4',d4.shape)
        # print('out',outputs.shape)
        return outputs

train_losses = []
train_losses_2 = []
train_losses_10 = []
train_losses_c = []
train_counter = []
test_losses = []
test_counter = []#[0,60000,120000,180000]#[i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train_classification(epoch, nclass, classifier, network, optimizer, train_loader):
    log_interval = 50
    classifier.train()
    network.eval()
    for batch_idx, (data, target) in enumerate(train_loader):
        # st()
        data = data.to(device)
        target = target.to(device)
        target = target%nclass
        optimizer.zero_grad()
        attacked_data = network(data)
        # st()
        output = classifier(attacked_data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            # torch.save(classifier.state_dict(), f'./results/model_{nclass}_pertubed.pth')
            # torch.save(optimizer.state_dict(), f'./results/optimizer_{nclass}_pertubed.pth')

def test_classification(nclass, classifier, network, test_loader):
    classifier.eval()
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
        #   st()
            data = data.to(device)
            target = target.to(device)
            target = target%nclass
            output = classifier(network(data))
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    # test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    # print(target,'\n',output)
    # st()
    return test_loss, correct / len(test_loader.dataset)

def visualize(classifier2, classifier10, network, test_loader):
    map = {1:'odd', 0:'even'}
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print('>>data shape: ', example_data.shape)
    example_data=example_data.to(device)
    example_targets=example_targets.to(device)
    with torch.no_grad():
        attacked_data = network(example_data)
        output_old = classifier10(example_data)
        output_new = classifier10(attacked_data)
        parity_old = classifier2(example_data)
        parity_new = classifier2(attacked_data)
    fig = plt.figure()
    for i in range(6):
        label_old = output_old.data.max(1, keepdim=True)[1][i].item()
        label_new = output_new.data.max(1, keepdim=True)[1][i].item()
        plt.subplot(3,6,i+1)
        plt.tight_layout()
        plt.imshow(example_data.cpu()[i][0], cmap='gray', interpolation='none')
        plt.title(f"{map[parity_old.data.max(1, keepdim=True)[1][i].item()]}: {label_old}")
        plt.xticks([])
        plt.yticks([])
        plt.subplot(3,6,i+7)
        plt.tight_layout()
        # st()
        plt.imshow(torch.stack((attacked_data.cpu()[i][0]-example_data.cpu()[i][0], example_data.cpu()[i][0]-attacked_data.cpu()[i][0], torch.zeros_like(attacked_data.cpu()[i][0])),2), cmap='bwr', interpolation='none')
        plt.title("")
        plt.xticks([])
        plt.yticks([])
        plt.subplot(3,6,i+13)
        plt.tight_layout()
        plt.imshow(attacked_data.cpu()[i][0], cmap='gray', interpolation='none')
        plt.text(23, 0,f"{map[parity_new.data.max(1, keepdim=True)[1][i].item()]}: ",color = 'green' if parity_new.data.max(1, keepdim=True)[1][i].item()==parity_old.data.max(1, keepdim=True)[1][i].item() else 'red',ha ='right', va="bottom")
        plt.text(23.90, 0,f"{label_new}",color = 'green' if label_new==label_old else 'red', va="bottom")
        plt.xticks([])
        plt.yticks([])
    plt.savefig(f'visualize.png',bbox_inches='tight')

def train_unet(epoch, classifier2, classifier10, network, optimizer, train_loader, alpha=100, beta=0.2):
    log_interval = 50
    network.train()
    classifier2.eval()
    classifier10.eval()
    a,b=8,8
    for batch_idx, (data, target) in enumerate(train_loader):
        beta+=0.0001
        a-=0.0001
        b-=0.0001
        # st()
        data = data.to(device)
        target = target.to(device)
        target2 = target%2
        target10 = target%10
        optimizer.zero_grad()
        attacked_data = network(data)
        # st()
        output2 = classifier2(attacked_data)
        output10 = classifier10(attacked_data)
        loss2 = F.nll_loss(output2, target2)
        loss10 = F.nll_loss(output10, target10)
        constrain = F.mse_loss(attacked_data, data)
        # st()
        # loss = alpha*loss2/loss10 + beta*constrain
        loss = loss2+max(0,30-loss10)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}\t>loss2: {loss2.item():.6f}-loss10: {loss10.item():.6f}-constrain: {constrain:.6f}')
            train_losses_2.append(loss2.item())
            train_losses_10.append(loss10.item())
            train_losses_c.append(constrain.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(network.state_dict(), f'./results/model_pertub_try.pth')
            torch.save(optimizer.state_dict(), f'./results/optimizer_pertub_try.pth')

def test_unet(classifier2, classifier10, network, test_loader, alpha=3, beta=0.2):
    network.eval()
    classifier2.eval()
    classifier10.eval()
    test_loss = 0
    correct2_old = 0
    correct2_new = 0
    correct10_old = 0
    correct10_new = 0
    with torch.no_grad():
        for data, target in test_loader:
            #   st()
            data = data.to(device)
            target = target.to(device)
            target2 = target%2
            target10 = target%10
            attacked_data = network(data)
            # st()
            output2_old = classifier2(data)
            output10_old = classifier10(data)
            output2_new = classifier2(attacked_data)
            output10_new = classifier10(attacked_data)
            # test_loss += alpha*F.nll_loss(output2, target2, size_average=False).item() / F.nll_loss(output10, target10, size_average=False).item()
            pred2_old = output2_old.data.max(1, keepdim=True)[1]
            correct2_old += pred2_old.eq(target2.data.view_as(pred2_old)).sum()
            pred10_old = output10_old.data.max(1, keepdim=True)[1]
            correct10_old += pred10_old.eq(target10.data.view_as(pred10_old)).sum()
            pred2_new = output2_new.data.max(1, keepdim=True)[1]
            correct2_new += pred2_new.eq(target2.data.view_as(pred2_new)).sum()
            pred10_new = output10_new.data.max(1, keepdim=True)[1]
            correct10_new += pred10_new.eq(target10.data.view_as(pred10_new)).sum()
    # test_loss /= len(test_loader.dataset)
    # test_losses.append(test_loss)
    # print(f'\nTest set: Avg. loss: {test_loss:.4f}\n2Classification Accuracy: {correct2}/{len(test_loader.dataset)} ({100. * correct2 / len(test_loader.dataset):.0f}%)\n10Classification Accuracy: {correct10}/{len(test_loader.dataset)} ({100. * correct10 / len(test_loader.dataset):.0f}%)\n')
    print(f'Test set: Old2Classification Accuracy: {correct2_old}/{len(test_loader.dataset)} ({100. * correct2_old / len(test_loader.dataset):.0f}%)')
    print(f'Test set: New2Classification Accuracy: {correct2_new}/{len(test_loader.dataset)} ({100. * correct2_new / len(test_loader.dataset):.0f}%)')
    print(f'Test set: Old10Classification Accuracy: {correct10_old}/{len(test_loader.dataset)} ({100. * correct10_old / len(test_loader.dataset):.0f}%)')
    print(f'Test set: New10Classification Accuracy: {correct10_new}/{len(test_loader.dataset)} ({100. * correct10_new / len(test_loader.dataset):.0f}%)')
    # print(target,'\n',output)
    # st()


def main(dataset, step):
    """ Parameters """
    n_epochs = 3
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5

    """ Data """
    if dataset.lower() == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./files/', train=True, download=True,
                                        transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor()
                                        # torchvision.transforms.Normalize(
                                        #     (0.1307,), (0.3081,))
                                        ])),
            batch_size=batch_size_train, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./files/', train=False, download=True,
                                        transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor()
                                        # torchvision.transforms.Normalize(
                                        #     (0.1307,), (0.3081,))
                                        ])),
            batch_size=batch_size_test, shuffle=True)
    elif dataset.lower() == 'cifar':
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10('./files/', train=True, download=True,
                                        transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Resize(
                                            28)
                                        ])),
            batch_size=batch_size_train, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10('./files/', train=False, download=True,
                                        transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Resize(
                                            28)
                                        ])),
            batch_size=batch_size_test, shuffle=True)

    """ Models """
    if dataset.lower() == 'mnist':
        passnet = PassNet().to(device)
        classifier2 = Net(2).to(device)
        classifier10 = Net(10).to(device)
        network = UNet().to(device)
    elif dataset.lower() == 'cifar':
        passnet = PassNet().to(device)
        classifier2 = Net(2,c=3).to(device)
        classifier10 = Net(10,c=3).to(device)
        network = UNet(c=3).to(device)
    # nclass = 2
    # classifier = Net(nclass).to(device)
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                        momentum=momentum)
    # st()

    if step!=0:
        network_state_dict = torch.load(f'./results/{dataset}_model_{2}_clean.pth')
        classifier2.load_state_dict(network_state_dict)
        network_state_dict = torch.load(f'./results/{dataset}_model_{10}_clean.pth')
        classifier10.load_state_dict(network_state_dict)

    # test_classification(nclass, network, test_loader)
    # for epoch in range(1, n_epochs + 1):
    #     train_classification(epoch, nclass, network, optimizer, train_loader)
    #     test_classification(nclass, network, test_loader)
        #   st()

    test_accs = []

    
    def main_unet():
        test_unet(classifier2, classifier10, network, test_loader)
        visualize(classifier2, classifier10, network, test_loader)
        for epoch in range(1, n_epochs + 1):
            train_unet(epoch, classifier2, classifier10, network, optimizer, train_loader)
            test_unet(classifier2, classifier10, network, test_loader)
            
            fig = plt.figure()
            plt.plot(train_counter, train_losses_2, color='blue')
            plt.plot(train_counter, train_losses_10, color='green')
            plt.plot(train_counter, train_losses_c, color='red')
            # plt.scatter(test_counter, test_losses, color='red')
            plt.legend(['Parity Loss', 'Classification Loss', 'Constrain'], loc='upper right')
            plt.xlabel('number of training examples seen')
            plt.ylabel('negative log likelihood loss')
            plt.savefig(f'loss.png',bbox_inches='tight')
            visualize(classifier2, classifier10, network, test_loader)
    
    def main_classifier(nclass, clean=False):
        if dataset.lower() == 'mnist':
            classifier = Net(nclass).to(device)
            network = UNet().to(device)
        elif dataset.lower() == 'cifar':
            classifier = Net(nclass, c=3).to(device)
            network = UNet(c=3).to(device)
        optimizer = optim.SGD(classifier.parameters(), lr=learning_rate,
                        momentum=momentum)
        if clean: 
            network = PassNet().to(device)
        else:
            network_state_dict = torch.load(f'./results/model_pertub_try.pth')
            network.load_state_dict(network_state_dict)
        print(f'Tringing classifier for {nclass}-class task')
        test_loss, test_acc = test_classification(nclass, classifier, passnet, test_loader)
        test_accs.append(test_acc.item())
        test_losses.append(test_loss)
        for epoch in range(1, n_epochs + 1):
            train_classification(epoch, nclass, classifier, network, optimizer, train_loader)
            test_loss, test_acc = test_classification(nclass, classifier, passnet, test_loader)
            test_accs.append(test_acc.item())
            test_losses.append(test_loss)
        fig, ax = plt.subplots()
        ax2=ax.twinx()
        test_counter = [i*len(train_loader.dataset) for i in range(len(test_losses))]
        # st()
        ax.plot(train_counter, train_losses, color='blue')
        ax.scatter(test_counter, test_losses, color='red')
        ax2.plot(test_counter, test_accs, color='green')
        ax2.scatter(test_counter, test_accs, color='green')
        ax.legend(['Train Loss', 'Test Loss'])
        ax2.legend(['Test Acc'])
        ax.set_xlabel('number of training examples seen')
        ax.set_ylabel('negative log likelihood loss')
        # ax.ylim([0,5])
        # ax.axis(ymin=0,ymax=3)
        ax2.set_ylabel('accuracy')
        # plt.ylim([0,4])
        plt.savefig(f'loss.png',bbox_inches='tight')
        if clean:
            torch.save(classifier.state_dict(), f'./results/{dataset}_model_{nclass}_clean.pth')
            # torch.save(optimizer.state_dict(), f'./results/optimizer_{nclass}_clean.pth')
        else:
            torch.save(classifier.state_dict(), f'./results/{dataset}_model_{nclass}_pertubed.pth')
        
        return test_accs

    if step==0:
        main_classifier(10, clean=True)
        main_classifier(2, clean=True)
    elif step==1:
        main_unet()
    elif step==2:
        main_classifier(10)
        main_classifier(2)
main('cifar', step=0)