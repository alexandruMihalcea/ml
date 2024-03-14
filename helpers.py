from typing import Tuple, List, Dict
from matplotlib import pyplot as plt
import torch
import torchvision


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """
    Performs training step with model on the given dataloader
    :param model: the model to train
    :param data_loader: the dataloader to use for training step
    :param loss_fn: the loss function
    :param optimizer: the optimizer
    :param accuracy_fn: accuracy measurement function
    :param device: the device on which to do the training, should be the device the model is on
    :return: the train loss and accuracy
    """

    train_loss, train_acc = 0, 0

    model.train()

    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        # do the forward pass
        y_pred = model(X)
        #calculate the loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)

        train_acc += (y_pred_class == y).sum().item() / len(y_pred)
        #optimizer zero grad
        optimizer.zero_grad()
        # loss backward
        loss.backward()
        #optimizer step
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """
    Performs the test step on the model, using the given dataloader
    :param model: the model to do the test step on
    :param data_loader: the dataloader containing the test data
    :param loss_fn: the loss function
    :param accuracy_fn: the accuracy function
    :param device: the device, should be the same device the model is on
    :return:
    """
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)

            test_pred = model(X)

            test_loss += loss_fn(test_pred, y).item()

            y_pred_class = torch.argmax(torch.softmax(test_pred, dim=1), dim=1)
            test_acc += (y_pred_class == y).sum().item() / len(test_pred)

        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        return test_loss, test_acc


def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary"""
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_accuracy"]
    test_accuracy = results["test_accuracy"]

    epochs = range(len(results["train_loss"]))

    # setup a plot :
    plt.figure(figsize=(15, 7))

    # plot the loss :
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")

    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot the accuracy :
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

def make_pred(image_path, model, classes, transforms, device):
    # Load in the custom image and convert to float32 :
    custom_image = torchvision.io.read_image(str(image_path)).type(torch.float32) / 255
    custom_image_transformed = transforms(custom_image)

    model.eval()
    logits = model(custom_image_transformed.unsqueeze(dim=0).to(device))
    y_pred_class = classes[torch.argmax(torch.softmax(logits, dim=1), dim=1)]
    y_pred_prob = torch.max(torch.softmax(logits, dim=1), dim=1).values

    plt.imshow(custom_image.permute(1, 2, 0))
    plt.title(f"{y_pred_class} | Probability: {float(y_pred_prob) : .2f}")
    plt.axis(False)
    return y_pred_class, float(y_pred_prob)

#%%
