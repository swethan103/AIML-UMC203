import argparse
import os
import pickle

import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms

import nice


def train(flow, trainloader, optimizer, device, lambda_model, lambda_mu, lambda_log_sigma):
    flow.train()
    epoch_total_loss = 0.0
    epoch_nll_loss = 0.0
    epoch_map_penalty = 0.0

    for inputs, _ in trainloader:
        inputs = inputs.view(
            inputs.shape[0],
            inputs.shape[1] * inputs.shape[2] * inputs.shape[3]
        ).to(device)

        optimizer.zero_grad()

        log_prob = flow(inputs)
        nll_loss = -log_prob.mean()

        map_penalty = flow.map_penalty(
            lambda_model=lambda_model,
            lambda_mu=lambda_mu,
            lambda_log_sigma=lambda_log_sigma
        )

        total_loss = nll_loss + map_penalty

        total_loss.backward()
        optimizer.step()

        epoch_total_loss += total_loss.item()
        epoch_nll_loss += nll_loss.item()
        epoch_map_penalty += map_penalty.item()

    num_batches = len(trainloader)
    return {
        "total_loss": epoch_total_loss / num_batches,
        "nll_loss": epoch_nll_loss / num_batches,
        "map_penalty": epoch_map_penalty / num_batches,
    }


def test(flow, testloader, device, lambda_model, lambda_mu, lambda_log_sigma):
    flow.eval()
    epoch_total_loss = 0.0
    epoch_nll_loss = 0.0
    epoch_map_penalty = 0.0

    with torch.no_grad():
        for inputs, _ in testloader:
            inputs = inputs.view(
                inputs.shape[0],
                inputs.shape[1] * inputs.shape[2] * inputs.shape[3]
            ).to(device)

            log_prob = flow(inputs)
            nll_loss = -log_prob.mean()

            map_penalty = flow.map_penalty(
                lambda_model=lambda_model,
                lambda_mu=lambda_mu,
                lambda_log_sigma=lambda_log_sigma
            )

            total_loss = nll_loss + map_penalty

            epoch_total_loss += total_loss.item()
            epoch_nll_loss += nll_loss.item()
            epoch_map_penalty += map_penalty.item()

    num_batches = len(testloader)
    return {
        "total_loss": epoch_total_loss / num_batches,
        "nll_loss": epoch_nll_loss / num_batches,
        "map_penalty": epoch_map_penalty / num_batches,
    }


def save_samples(flow, epoch, sample_shape, sample_dir, num_samples=100):
    flow.eval()
    with torch.no_grad():
        samples = flow.sample(num_samples)

        a, b = samples.min(), samples.max()
        samples = (samples - a) / (b - a + 1e-10)

        samples = samples.view(
            -1, sample_shape[0], sample_shape[1], sample_shape[2]
        )

        grid = torchvision.utils.make_grid(samples, nrow=10)
        save_path = os.path.join(sample_dir, f"epoch_{epoch:04d}.png")
        torchvision.utils.save_image(grid, save_path)


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs("samples", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    sample_shape = [1, 28, 28]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.0,)),
        transforms.Lambda(
            lambda x: x + torch.zeros_like(x).uniform_(0.0, 1.0 / 256.0)
        ),
    ])

    if args.dataset == "mnist":
        trainset = torchvision.datasets.MNIST(
            root="./data/MNIST",
            train=True,
            download=True,
            transform=transform
        )
        testset = torchvision.datasets.MNIST(
            root="./data/MNIST",
            train=False,
            download=True,
            transform=transform
        )
    elif args.dataset == "fashion-mnist":
        trainset = torchvision.datasets.FashionMNIST(
            root="./data/FashionMNIST",
            train=True,
            download=True,
            transform=transform
        )
        testset = torchvision.datasets.FashionMNIST(
            root="./data/FashionMNIST",
            train=False,
            download=True,
            transform=transform
        )
    else:
        raise ValueError("Dataset not implemented")

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    full_dim = 28 * 28

    flow = nice.NICE(
        prior=args.prior,
        coupling=args.coupling,
        coupling_type=args.coupling_type,
        in_out_dim=full_dim,
        mid_dim=args.mid_dim,
        hidden=args.hidden,
        device=device
    ).to(device)

    optimizer = torch.optim.Adam(flow.parameters(), lr=args.lr)

    train_total_history = []
    train_nll_history = []
    train_map_history = []

    test_total_history = []
    test_nll_history = []
    test_map_history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train(
            flow,
            trainloader,
            optimizer,
            device,
            args.lambda_model,
            args.lambda_mu,
            args.lambda_log_sigma
        )

        test_metrics = test(
            flow,
            testloader,
            device,
            args.lambda_model,
            args.lambda_mu,
            args.lambda_log_sigma
        )

        train_total_history.append(train_metrics["total_loss"])
        train_nll_history.append(train_metrics["nll_loss"])
        train_map_history.append(train_metrics["map_penalty"])

        test_total_history.append(test_metrics["total_loss"])
        test_nll_history.append(test_metrics["nll_loss"])
        test_map_history.append(test_metrics["map_penalty"])

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Total: {train_metrics['total_loss']:.4f} | "
            f"Train NLL: {train_metrics['nll_loss']:.4f} | "
            f"Train MAP: {train_metrics['map_penalty']:.4f} | "
            f"Test Total: {test_metrics['total_loss']:.4f} | "
            f"Test NLL: {test_metrics['nll_loss']:.4f} | "
            f"Test MAP: {test_metrics['map_penalty']:.4f}"
        )

        if epoch % 10 == 0:
            save_samples(
                flow,
                epoch,
                sample_shape,
                sample_dir="samples",
                num_samples=args.sample_size
            )

        if epoch % 50 == 0:
            ckpt_path = os.path.join("checkpoints", f"nice_epoch_{epoch:04d}.pt")
            torch.save(flow.state_dict(), ckpt_path)

    final_ckpt = os.path.join("checkpoints", "nice_final.pt")
    torch.save(flow.state_dict(), final_ckpt)

    history = {
        "train_total": train_total_history,
        "train_nll": train_nll_history,
        "train_map": train_map_history,
        "test_total": test_total_history,
        "test_nll": test_nll_history,
        "test_map": test_map_history,
    }

    with open("training_history.pkl", "wb") as f:
        pickle.dump(history, f)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.epochs + 1), train_total_history, label="Train Total Loss")
    plt.plot(range(1, args.epochs + 1), test_total_history, label="Test Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("NICE with Learned Prior + MAP over Model and Prior")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_total.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.epochs + 1), train_nll_history, label="Train NLL")
    plt.plot(range(1, args.epochs + 1), test_nll_history, label="Test NLL")
    plt.xlabel("Epoch")
    plt.ylabel("NLL")
    plt.title("Negative Log-Likelihood")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_nll.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.epochs + 1), train_map_history, label="Train MAP Penalty")
    plt.plot(range(1, args.epochs + 1), test_map_history, label="Test MAP Penalty")
    plt.xlabel("Epoch")
    plt.ylabel("Penalty")
    plt.title("MAP Penalty")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_map_penalty.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--prior", type=str, default="learned_gaussian")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--coupling_type", type=str, default="additive")
    parser.add_argument("--coupling", type=int, default=4)
    parser.add_argument("--mid_dim", type=int, default=1000)
    parser.add_argument("--hidden", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--lambda_model", type=float, default=1e-5)
    parser.add_argument("--lambda_mu", type=float, default=1e-4)
    parser.add_argument("--lambda_log_sigma", type=float, default=1e-4)

    args = parser.parse_args()
    main(args)