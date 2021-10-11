import torch
import numpy as np
import logging
import warnings
import argparse

warnings.filterwarnings('ignore')

from dataprocessing import BreastCancerDemoDataset
import networks.sklearn_network as demonet



def main(args):
    print("---------------------LOADING DATA---------------------")
    training_data = BreastCancerDemoDataset('train')
    dataloader = torch.utils.data.DataLoader(training_data, batch_size=args.batchsize, shuffle=args.doshuffle, drop_last=True)

    '''
    Set up val and test data
    '''
    validation_data = BreastCancerDemoDataset('val')
    test_data = BreastCancerDemoDataset('test')
    validation_dataloader = torch.utils.data.DataLoader(validation_data, batch_size=args.val_batchsize, shuffle=args.doshuffle)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batchsize, shuffle=args.doshuffle,
                                                  drop_last=False)

    print("----DATA LOADED----")

    torch.random.manual_seed(args.seed)
    predictor = demonet.DemoNet()

    lossfunction = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(params=predictor.parameters(), lr=1e-4)

    print("------------------TRAINING STARTING-------------------")
    print("----------------TRAINING FOR " + str(args.n_epochs) + " EPOCHS-----------------")

    for epoch in range(args.n_epochs):
        predictor.train()
        for ii, sample_batch in enumerate(dataloader):
            optimizer.zero_grad()
            data = torch.tensor(np.asarray(sample_batch[0]), dtype=torch.float32)
            labels = torch.tensor(np.asarray(sample_batch[1]), dtype=torch.float32)
            predictions = predictor(data)

            loss = lossfunction(predictions, labels)
            loss.backward()

            optimizer.step()

        val_errors = 0.0
        loss = 0.0
        predictor.eval()
        for ii, sample_batch in enumerate(validation_dataloader):
            data = torch.tensor(np.asarray(sample_batch[0]), dtype=torch.float32)
            targets = torch.tensor(np.asarray(sample_batch[1]), dtype=torch.float32)

            net_predictions = predictor(data).squeeze()

            loss = lossfunction(net_predictions, targets)
            sum_of_errors = torch.sum(torch.abs(torch.round(net_predictions) - targets))
            val_errors += sum_of_errors.detach().numpy()

        validation_accuracy = str(100. - 100.0 * val_errors / len(validation_data))
        validation_loss = str(loss.detach().numpy())
        print("Epoch: " + str(epoch).zfill(2) + ", Network Validation Accuracy: " + validation_accuracy +
              "%, Current Validation Loss: " + validation_loss)

    test_errors = 0.0
    running_loss = 0.0
    max_ii = 0
    for ii, sample_batch in enumerate(test_dataloader):
        data = torch.tensor(np.asarray(sample_batch[0]), dtype=torch.float32)
        targets = torch.tensor(np.asarray(sample_batch[1]), dtype=torch.float32)

        net_predictions = predictor(data).squeeze()

        loss = lossfunction(net_predictions, targets)
        running_loss += loss.detach().numpy()

        sum_of_errors = torch.sum(torch.abs(torch.round(net_predictions) - targets)).detach().numpy()
        test_errors += sum_of_errors
        max_ii = ii
    test_loss = str(running_loss / (max_ii + 1))
    test_accuracy = str(100. - 100.0 * test_errors / len(test_data))

    print("------------------TRAINING COMPLETED------------------")
    print("Network Test Accuracy: " + test_accuracy + "%, Test Cross Entropy Loss: " + test_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", default=100, type=int)
    parser.add_argument("--val_batchsize", default=99, type=int)
    parser.add_argument("--test_batchsize", default=16, type=int)
    parser.add_argument("--doshuffle", action="store_true")
    parser.add_argument("--n_epochs", default=3, type=int)
    parser.add_argument("--seed", default=11, type=int)

    args = parser.parse_args()
    main(args)