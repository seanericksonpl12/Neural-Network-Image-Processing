import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    if(training):
        train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
        loader = torch.utils.data.DataLoader(train_set, batch_size = 50)
        return loader

    test_set = datasets.MNIST('./data', train=False, transform=transform)
    loader = torch.utils.data.DataLoader(test_set, batch_size = 50)
    return loader


def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:

        An untrained neural network model
    """
    model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
    )
    return model



def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    for epoch in range(T):
        running_loss = 0.0
        correct = 0
        for image, label in train_loader:
            opt.zero_grad()

            predict = model(image)
            for i, x in enumerate(predict):
                if(torch.argmax(x).item() == label[i].item()):
                    correct += 1
            
            loss = criterion(predict, label)
            loss.backward()
            opt.step()
            running_loss += (loss.item() * 50)
        rtrn = "Train Epoch: "+str(epoch) +"   Accuracy: "+str(correct)+"/60000("+str('{:.2f}'.format((correct/60000) * 100))+"%) Loss: "+str('{:.3f}'.format(running_loss/60000))
        print(rtrn)


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, labels in test_loader:

            predict = model(data)
            for i, x in enumerate(predict):
                if(torch.argmax(x).item() == labels[i].item()):
                    correct += 1

            loss = criterion(predict, labels)
            running_loss += (loss.item() * 50)
    rtrn = "Accuracy: "+str('{:.2f}'.format((correct/10000) * 100))+"%"
    loss_str = "Average loss: " + str('{:.4f}'.format(running_loss/10000))
    if(show_loss):
        print(loss_str)
        print(rtrn)
    else:
        print(rtrn)
    


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    class_names = {0:'zero', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five', 6:'six', 7:'seven', 8:'eight', 9:'nine'}
    
    logits = model(test_images[index])
    prob = F.softmax(logits, dim=1)
    sorted, indices = torch.topk(prob, 3)
    rtrn1 = class_names[indices[0][0].item()] + ": " + str('{:.2f}'.format(sorted[0][0].item()*100)) + "%"
    rtrn2 = class_names[indices[0][1].item()] + ": " + str('{:.2f}'.format(sorted[0][1].item()*100)) + "%"
    rtrn3 = class_names[indices[0][2].item()] + ": " + str('{:.2f}'.format(sorted[0][2].item()*100)) + "%"
    print(rtrn1)
    print(rtrn2)
    print(rtrn3)
   
    


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
