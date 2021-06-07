import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time
import os

# Zeyun Wu
# Feb 27 2020
# script for generating adv example on testing image with trained CNN in pytorch
# overlap with ECE 269 projects where we also need to generate adversarial examples using FGSM


# generate perturbed image using FGSM
def fgsm_perturb(image, epsilon, gradient):
    '''
    image: input image; should be 3 channel torch
    epsilon: how much to perturb
    gradient: gradient of loss w.r.t input image from the network; should be same shape as image
    '''
    
    if epsilon == 0:
        return image
    
    sign_data_grad = gradient.sign()
    perturbed_image = image + epsilon*sign_data_grad
    
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image


def test(model, device, test_loader, epsilon, visual=True):

    # make sure model is in evaluation mode
    _ = model.eval()
    
    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # calculate the loss
        loss = F.nll_loss(output, target)

        # zero all existing gradients
        model.zero_grad()

        # calculate gradients of model in backward pass
        loss.backward()

        # collect datagrad
        data_grad = data.grad.data

        # call FGSM attack
        perturbed_data = fgsm_perturbe(data, epsilon, data_grad)

        # re-classify the perturbed image
        output = model(perturbed_data)

        # check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            if visual:
                # special case for saving 0 epsilon examples
                if (epsilon == 0) and (len(adv_examples) < 5):
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            if visual:
                # save some adv examples for visualization later
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples