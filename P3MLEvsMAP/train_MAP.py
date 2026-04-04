"""Training procedure for NICE.
"""

import argparse
import torch
import torchvision
from torchvision import transforms
from collections import defaultdict
from tqdm import trange
import matplotlib.pyplot as plt
import nice_MAP
import pickle


def train(flow, trainloader, optimizer, epoch, device,lambda_map):
    flow.train()  # set to training mode
    epoch_total_loss = 0.
    epoch_nll = 0.
    epoch_penalty = 0.
    for inputs, _ in trainloader:
        inputs = inputs.view(inputs.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[3]).to(
            device)  # change  shape from BxCxHxW to Bx(C*H*W)
        # TODO Fill in

        # zero the grads
        optimizer.zero_grad()

        # forward
        # forward
        nll = -flow(inputs).mean()

        l2_penalty = 0.
        for param in flow.parameters():
          l2_penalty = l2_penalty + torch.sum(param ** 2)

        total_loss = nll + lambda_map * l2_penalty

        
        # backward
        total_loss.backward()

        # step
        optimizer.step()

        epoch_total_loss += total_loss.item()
        epoch_nll += nll.item()
        epoch_penalty += l2_penalty.item()

    num_batches = len(trainloader)
    return (
        epoch_total_loss / num_batches,
        epoch_nll / num_batches,
        epoch_penalty / num_batches,
    )


def test(flow, testloader, filename, epoch, sample_shape, device, should_sample: bool = False):
    flow.eval()  # set to inference mode
    epoch_loss = 0.
    with torch.no_grad():
        if should_sample:
            samples = flow.sample(100)
            a, b = samples.min(), samples.max()
            samples = (samples - a) / (b - a + 1e-10)
            samples = samples.view(-1, sample_shape[0], sample_shape[1], sample_shape[2])
            torchvision.utils.save_image(torchvision.utils.make_grid(samples),
                                         './samples/' + filename + 'epoch%d.png' % epoch)
        # TODO full in

        for inputs, _ in testloader:
            inputs = inputs.view(inputs.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[3]).to(
                device)  # change  shape from BxCxHxW to Bx(C*H*W)
            # forward
            loss = - flow(inputs).mean()

            epoch_loss += loss.item()

        return epoch_loss / len(testloader)


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sample_shape = [1, 28, 28]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.,)),
        transforms.Lambda(lambda x: x + torch.zeros_like(x).uniform_(0., 1. / 256.))  # dequantization
    ])

    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST',
                                              train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size, shuffle=True, num_workers=0)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
                                             train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=0)
    elif args.dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root='~/torch/data/FashionMNIST',
                                                     train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
                                                    train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=2)
    else:
        raise ValueError('Dataset not implemented')
    
    lambda_tag = str(args.lambda_map).replace('.', 'p').replace('-', 'm')
    run_name = (
        f"{args.dataset}_"
        f"{args.coupling_type}_"
        f"coupling{args.coupling}_"
        f"mid{args.mid_dim}_"
        f"hidden{args.hidden}_"
        f"lr{args.lr}_"
        f"lmap{lambda_tag}"
    )

    model_save_filename = f"{run_name}.pt"

    full_dim = 28 * 28
    flow = nice_MAP.NICE(
        prior=args.prior,
        coupling=args.coupling,
        coupling_type=args.coupling_type,
        in_out_dim=full_dim,
        mid_dim=args.mid_dim,
        hidden=args.hidden,
        device=device).to(device)

    optimizer = torch.optim.Adam(
        flow.parameters(), lr=args.lr)

    # TODO fill in

    # history to plot
    train_total_history = []
    train_nll_history = []
    train_penalty_history = []
    test_nll_history = []

    for epoch in range(args.epochs):
        train_total, train_nll, train_penalty = train(
            flow, trainloader, optimizer, epoch, device, args.lambda_map
        )

        train_total_history.append(train_total)
        train_nll_history.append(train_nll)
        train_penalty_history.append(train_penalty)

        test_nll = test(flow, testloader, model_save_filename, epoch, sample_shape, device)
        test_nll_history.append(test_nll)

        print(
            f"Epoch: {epoch} / {args.epochs} | "
            f"Train total = {train_total:.4f} | "
            f"Train NLL = {train_nll:.4f} | "
            f"Train penalty = {train_penalty:.4f} | "
            f"Test NLL = {test_nll:.4f}"
        )

    # sample
    test(flow, testloader, model_save_filename, args.epochs,
         sample_shape, device, should_sample=True)
    
    train_metrics = {
        "train_total_history": train_total_history,
        "train_nll_history": train_nll_history,
        "train_penalty_history": train_penalty_history,
        "lambda_map": args.lambda_map,
        "dataset": args.dataset,
        "coupling_type": args.coupling_type,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
    }

    test_metrics = {
        "test_nll_history": test_nll_history,
        "lambda_map": args.lambda_map,
        "dataset": args.dataset,
        "coupling_type": args.coupling_type,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
    }

        
    # save history
    train_filename = f"train_metrics_{run_name}.pkl"
    test_filename = f"test_metrics_{run_name}.pkl"
    plot_filename = f"loss_plot_{run_name}.png"

    with open(train_filename, "wb") as f:
        pickle.dump(train_metrics, f)

    with open(test_filename, "wb") as f:
        pickle.dump(test_metrics, f)

    
    plt.figure(figsize=(10, 6))
    plt.plot(range(args.epochs), train_nll_history, label='Train NLL', marker='o')
    plt.plot(range(args.epochs), test_nll_history, label='Test NLL', marker='x')

    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Negative Log-Likelihood', fontsize=12)
    plt.title(f'NLL over Epochs ({run_name})', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_filename)

    plt.figure(figsize=(10, 6))
    plt.plot(range(args.epochs), train_total_history, label='Train Total MAP Loss', marker='o')
    plt.plot(range(args.epochs), train_penalty_history, label='Train L2 Penalty', marker='x')

    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss / Penalty', fontsize=12)
    plt.title(f'MAP Training Components ({run_name})', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"map_components_{run_name}.png")

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='mnist')
    parser.add_argument('--prior',
                        help='latent distribution.',
                        type=str,
                        default='logistic')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=50)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)
    parser.add_argument('--coupling-type',
                        help='.',
                        type=str,
                        default='additive')
    parser.add_argument('--coupling',
                        help='.',
                        # type=int,
                        default=4)
    parser.add_argument('--mid-dim',
                        help='.',
                        type=int,
                        default=1000)
    parser.add_argument('--hidden',
                        help='.',
                        type=int,
                        default=5)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)
    parser.add_argument('--lambda-map',
                    help='strength of MAP parameter prior.',
                    type=float,
                    default=1e-5)

    args = parser.parse_args()
    main(args)
    