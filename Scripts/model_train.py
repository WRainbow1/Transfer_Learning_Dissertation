import numpy as np
import matplotlib.pyplot as plt
from models import (MNIST_Feature_Extractor, 
                    MNIST_Label_Predictor, 
                    MNIST_Domain_Classifier,
                    SVHN_Feature_Extractor, 
                    SVHN_Label_Predictor, 
                    SVHN_Domain_Classifier)
from scipy.io import loadmat
import sklearn

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
 
from tensorboardX import SummaryWriter

if __name__ == "__main__":

    writer = SummaryWriter(comment="-")

    s_dm = loadmat(r'C:\Users\willr\Documents\University\FYP\Model\Data\SVHN\SVHN_processed.mat')
    t_dm = loadmat(r'C:\Users\willr\Documents\University\FYP\Model\Data\MNIST\MNIST_processed.mat')

    batch_len = len(s_dm["X_batches"][0]) if \
                len(s_dm["X_batches"][0]) < len(t_dm["X_batches"][0]) else \
                len(t_dm["X_batches"][0])

    s_dm_X = s_dm["X_batches"][0][0:int(4*batch_len/5)]
    s_dm_y = s_dm["y_batches"][0][0:int(4*batch_len/5)]
    s_dm_d = s_dm["d_batches"][0][0:int(4*batch_len/5)]

    s_dm_val_X = s_dm["X_batches"][0][int(4*batch_len/5):batch_len]
    s_dm_val_y = s_dm["y_batches"][0][int(4*batch_len/5):batch_len]
    s_dm_val_d = s_dm["d_batches"][0][int(4*batch_len/5):batch_len]

    t_dm_X = t_dm["X_batches"][0][0:int(4*batch_len/5)]
    t_dm_y = t_dm["y_batches"][0][0:int(4*batch_len/5)]
    t_dm_d = t_dm["d_batches"][0][0:int(4*batch_len/5)]

    t_dm_val_X = t_dm["X_batches"][0][int(4*batch_len/5):batch_len]
    t_dm_val_y = t_dm["y_batches"][0][int(4*batch_len/5):batch_len]
    t_dm_val_d = t_dm["d_batches"][0][int(4*batch_len/5):batch_len]
 
    device = torch.device("cuda")
    INPUT_SHAPE = s_dm_X[0].shape[1:]
    N_CLASSES = 10
    LEARNING_RATE = 1e-4
    MODEL_ARCHITECTURE = "SVHN"
    EPOCHS = 400
    dropout_prob = 0.2

    epoch_list = []
    label_loss_list = []
    domain_loss_list = []
    source_val_acc_list = []
    target_val_acc_list = []

    if MODEL_ARCHITECTURE == "MNIST":
        feature_extractor = MNIST_Feature_Extractor(INPUT_SHAPE).to(device)
        label_predictor = MNIST_Label_Predictor(N_CLASSES).to(device)
        domain_classifier = MNIST_Domain_Classifier().to(device)
    
    elif MODEL_ARCHITECTURE == "SVHN":
        feature_extractor = SVHN_Feature_Extractor(INPUT_SHAPE, p=dropout_prob).to(device)
        label_predictor = SVHN_Label_Predictor(N_CLASSES, p=dropout_prob).to(device)
        domain_classifier = SVHN_Domain_Classifier(p=dropout_prob).to(device)

    feature_optimiser = optim.Adam(feature_extractor.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    label_optimiser = optim.Adam(label_predictor.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    domain_optimiser = optim.Adam(domain_classifier.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) #, momentum=0.9

    label_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCELoss()

    for epoch in range(EPOCHS):

        p = float(epoch)/EPOCHS
        lambd = 2. / (1. + np.exp(-10. * p)) - 1
        domain_classifier.set_lambda(lambd)

        for i, _ in enumerate(s_dm_X):
            feature_extractor.is_training(True)
            label_predictor.is_training(True)
            domain_classifier.is_training(True)

            ### Domain Classifier Training ###
            feature_optimiser.zero_grad()
            domain_optimiser.zero_grad()

            X_batch = np.concatenate((s_dm_X[i], t_dm_X[i]))
            d_batch = np.concatenate((s_dm_d[i], t_dm_d[i]))

            X_batch, d_batch = sklearn.utils.shuffle(X_batch, d_batch)
            X_batch, d_batch = X_batch[0:int(len(X_batch)/2)], d_batch[0:int(len(d_batch)/2)]
            X_batch = torch.from_numpy(X_batch).to(device, dtype=torch.float)
            d_batch = torch.from_numpy(d_batch).to(device, dtype=torch.float)

            domain_features = feature_extractor(X_batch)
            domain_output = domain_classifier(domain_features)

            domain_loss = domain_criterion(domain_output, d_batch)
            domain_loss.backward()

            feature_optimiser.step()
            domain_optimiser.step()

            ### Label Predictor Training ###
            feature_optimiser.zero_grad()
            label_predictor.zero_grad()

            s_dm_X_batch = torch.from_numpy(s_dm_X[i]).to(device, dtype=torch.float)
            s_dm_y_batch = torch.from_numpy(s_dm_y[i][:,0]).to(device, dtype=torch.long)

            s_dm_features = feature_extractor(s_dm_X_batch)
            s_dm_output = label_predictor(s_dm_features)

            label_loss = label_criterion(s_dm_output, s_dm_y_batch)
            label_loss.backward()
            
            feature_optimiser.step()
            label_optimiser.step()

        ### Validation ###
        sval_correct = 0
        sval_total = 0
        tval_correct = 0
        tval_total = 0
        domain_correct = 0
        domain_total = 0
        for i, _ in enumerate(s_dm_val_X):
            feature_extractor.is_training(False)
            label_predictor.is_training(False)
            domain_classifier.is_training(False)
            ### Source ###
            val_X_batch = torch.from_numpy(s_dm_val_X[i]).to(device, dtype=torch.float)
            val_y_batch = torch.from_numpy(s_dm_val_y[i]).to(device, dtype=torch.long)

            s_dm_features = feature_extractor(val_X_batch)
            s_dm_output = label_predictor(s_dm_features)

            max_vals, max_indices = torch.max(s_dm_output,1)
            sval_correct += (max_indices.unsqueeze(1) == val_y_batch).sum()
            sval_total += s_dm_output.shape[0]

            ### Target ###
            val_X_batch = torch.from_numpy(t_dm_val_X[i]).to(device, dtype=torch.float)
            val_y_batch = torch.from_numpy(t_dm_val_y[i]).to(device, dtype=torch.float)

            t_dm_features = feature_extractor(val_X_batch)
            t_dm_output = label_predictor(t_dm_features)

            max_vals, max_indices = torch.max(t_dm_output,1)
            tval_correct += (max_indices.unsqueeze(1) == val_y_batch).sum()
            tval_total += t_dm_output.shape[0]

            ### Domain ###
            X_batch = np.concatenate((s_dm_val_X[i], t_dm_val_X[i]))
            d_batch = np.concatenate((s_dm_val_d[i], t_dm_val_d[i]))

            X_batch, d_batch = sklearn.utils.shuffle(X_batch, d_batch)
            X_batch = torch.from_numpy(X_batch).to(device, dtype=torch.float)
            d_batch = torch.from_numpy(d_batch).to(device, dtype=torch.float)

            domain_features = feature_extractor(X_batch)
            domain_output = domain_classifier(domain_features)

            rounded_output = torch.round(domain_output)
            a = (rounded_output == d_batch).sum()
            domain_correct += (rounded_output == d_batch).sum()
            domain_total += domain_output.shape[0]            

        writer.add_scalar("label_loss", np.round(label_loss.item(),6), epoch)
        writer.add_scalar("domain_loss", np.round(domain_loss.item(),6), epoch)
        writer.add_scalar("source_val_accuracy", int(100*sval_correct/sval_total), epoch)
        writer.add_scalar("target_val_accuracy", int(100*tval_correct/tval_total), epoch)
        writer.add_scalar("domain_val_accuracy", int(100*domain_correct/domain_total), epoch)

        epoch_list.append(epoch)
        label_loss_list.append(np.round(label_loss.item(),6))
        domain_loss_list.append(np.round(domain_loss.item(),6))
        source_val_acc_list.append(int(100*sval_correct/sval_total))
        target_val_acc_list.append(int(100*tval_correct/tval_total))
    
    fig, axs = plt.subplots(1,2, figsize=(12,6))
    axs[0].plot(epoch_list, label_loss_list)
    axs[0].plot(epoch_list, domain_loss_list)
    axs[0].set_title("Label Classifier Loss")
    axs[0].set_xlabel("Training Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend(["Domain","Label"])
    axs[0].grid(True)

    axs[1].plot(epoch_list, source_val_acc_list)
    axs[1].plot(epoch_list, target_val_acc_list)
    axs[1].set_ylim(0,100)
    axs[1].set_title("Source and Target Domain Accuracy")
    axs[1].set_xlabel("Training Epoch")
    axs[1].set_ylabel("Accuracy (%)")
    axs[1].legend(["Source Domain","Target Domain"])
    axs[1].grid(True)

    plt.savefig("training_monitor_lin.png", dpi=200)

    fig, axs = plt.subplots(1,2, figsize=(12,6))
    axs[0].plot(epoch_list, label_loss_list)
    axs[0].plot(epoch_list, domain_loss_list)
    axs[0].set_yscale("log")
    axs[0].set_title("Label Classifier Loss")
    axs[0].set_xlabel("Training Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend(["Domain","Label"])
    axs[0].grid(True)

    axs[1].plot(epoch_list, source_val_acc_list)
    axs[1].plot(epoch_list, target_val_acc_list)
    axs[1].set_ylim(0,100)
    axs[1].set_title("Source and Target Domain Accuracy")
    axs[1].set_xlabel("Training Epoch")
    axs[1].set_ylabel("Accuracy (%)")
    axs[1].legend(["Source Domain","Target Domain"])
    axs[1].grid(True)

    plt.savefig("training_monitor_log.png", dpi=200)

    torch.save(feature_extractor.state_dict(), "feature_extractor_trained_weights.dat")
    torch.save(label_predictor.state_dict(), "label_predictor_trained_weights.dat")
    torch.save(domain_classifier.state_dict(), "domain_classifier_trained_weights.dat")