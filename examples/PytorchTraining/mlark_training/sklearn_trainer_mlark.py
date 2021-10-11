import torch
import numpy as np
import logging
import warnings

warnings.filterwarnings('ignore')

from dataprocessing import BreastCancerDemoDataset
import networks.sklearn_network as demonet

import maskedLark as mlark


def main(args):
    print("---------------------LOADING DATA---------------------")
    training_data = BreastCancerDemoDataset('train')
    mlark_dataset = mlark.MLarkDataset(training_data)  # Let the helper control data flow
    dataloader = torch.utils.data.DataLoader(mlark_dataset, batch_size=args.batchsize, shuffle=args.doshuffle, drop_last=True)

    '''
    Set up val and test data
    '''
    validation_data = BreastCancerDemoDataset('val')
    test_data = BreastCancerDemoDataset('test')
    validation_dataloader = torch.utils.data.DataLoader(validation_data, batch_size=args.val_batchsize, shuffle=args.doshuffle)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batchsize, shuffle=args.doshuffle,
                                                  drop_last=False)

    print("----DATA LOADED. SENDING " + str(args.batchsize) + " DATA POINTS PER BATCH----")

    print("-------------SETTING UP HELPER REQUESTS---------------")
    helper = mlark.Helper(mlark_dataset)
    helper.set_diff_privacy(mechanism='laplacian', norm_bound=args.grad_bound, epsilon=args.dp_eps)
    helper.set_aggregation_privacy(mechanism='standard', threshold=args.k_anonymity)
    helper.set_model_name('simplenet')
    helper.set_loss_fn('BCELoss', {})
    helper.set_helper_names(args.helpernames)
    helper.set_endpoints(args.helperendpoints)

    torch.random.manual_seed(11)
    predictor = demonet.DemoNet()

    lossfunction = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(params=predictor.parameters(), lr=1e-4)

    print("------------------TRAINING STARTING-------------------")
    print("----------------TRAINING FOR " + str(args.n_epochs) + " EPOCHS-----------------")

    for epoch in range(args.n_epochs):
        predictor.train()
        for ii, sample_batch in enumerate(dataloader):
            optimizer.zero_grad()

            helper.fetch_gradients(predictor)
            helper.backward()

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
    parser.add_argument("--helperendpoints", nargs='+', required=True, type=str, help="The addresses of the helper services.")
    parser.add_argument("--helpernames", nargs='+', required=True, type=str, help="The names of the helper services. There must be as many helpernames as helper endpoints.")

    parser.add_argument("--dp_eps", default=0.1, type=float, help="Epsilon budget for differential privacy.")
    parser.add_argument("--grad_bound", default=100, type=float, help="Bound to clip gradients to in the helper.")
    parser.add_argument("--k_anonymity", default=50, type=int)
    parser.add_argument("--batchsize", default=100, type=int)
    parser.add_argument("--val_batchsize", default=99, type=int)
    parser.add_argument("--test_batchsize", default=16, type=int)
    parser.add_argument("--doshuffle", action="store_true")
    parser.add_argument("--n_epochs", default=3, type=int)
    parser.add_argument("--seed", default=11, type=int)

    args = parser.parse_args()
    main(args)