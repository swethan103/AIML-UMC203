import argparse
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

import nice


def get_mnist_dataloaders(batch_size):
    transform = transforms.ToTensor()

    trainset = torchvision.datasets.MNIST(
        root="./data/MNIST",
        train=True,
        download=True,
        transform=transform,
    )
    testset = torchvision.datasets.MNIST(
        root="./data/MNIST",
        train=False,
        download=True,
        transform=transform,
    )

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return trainloader, testloader


def dequantize(x, data_bins):
    if data_bins <= 1.0:
        return x
    noise = torch.rand_like(x) / data_bins
    x = x + noise
    x = torch.clamp(x, 0.0, 1.0)
    return x


def save_checkpoint(flow, optimizer, epoch, batch_idx, global_step, checkpoint_dir, history):
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"nice_step_{global_step:06d}.pt"
    )

    torch.save(
        {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "global_step": global_step,
            "model_state_dict": flow.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "greedy_arms": flow.greedy_arm_indices(),
            "permutation_metadata": flow.get_permutation_metadata(),
        },
        checkpoint_path,
    )


def save_samples(flow, epoch, sample_shape, sample_dir, num_samples=100):
    flow.eval()
    with torch.no_grad():
        greedy_arms = flow.greedy_arm_indices()
        samples = flow.sample(num_samples, arm_indices=greedy_arms)

        a, b = samples.min(), samples.max()
        samples = (samples - a) / (b - a + 1e-10)

        samples = samples.view(
            -1, sample_shape[0], sample_shape[1], sample_shape[2]
        )

        grid = torchvision.utils.make_grid(samples, nrow=10)
        save_path = os.path.join(sample_dir, f"epoch_{epoch:04d}.png")
        torchvision.utils.save_image(grid, save_path)


def save_permutation_metadata(flow, out_path):
    metadata = flow.get_permutation_metadata()
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)


def save_permutation_summary_text(flow, out_path):
    metadata = flow.get_permutation_metadata()
    with open(out_path, "w") as f:
        for item in metadata:
            f.write(f"Arm {item['arm_index']:02d}: {item['name']}\n")
            f.write(f"Permutation: {item['permutation']}\n")
            f.write(f"Inverse:     {item['inverse_permutation']}\n")
            f.write("\n")


def plot_bandit_histograms(flow, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    counts = flow.bandit_counts.detach().cpu().numpy()
    num_layers, num_arms = counts.shape
    arm_labels = np.arange(num_arms)

    for layer_idx in range(num_layers):
        plt.figure(figsize=(16, 5))
        plt.bar(arm_labels, counts[layer_idx])
        plt.xlabel("Bandit Arm Index")
        plt.ylabel("Selection Count")
        plt.title(f"Layer {layer_idx}: Bandit Arm Usage Histogram")
        plt.xticks(arm_labels)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"layer_{layer_idx}_histogram.png"))
        plt.close()

    fig, axes = plt.subplots(num_layers, 1, figsize=(16, 4 * num_layers), squeeze=False)
    for layer_idx in range(num_layers):
        ax = axes[layer_idx, 0]
        ax.bar(arm_labels, counts[layer_idx])
        ax.set_title(f"Layer {layer_idx}: Bandit Arm Usage")
        ax.set_xlabel("Bandit Arm Index")
        ax.set_ylabel("Selection Count")
        ax.set_xticks(arm_labels)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_layers_histograms.png"))
    plt.close()

    greedy_arms = flow.greedy_arm_indices()
    with open(os.path.join(output_dir, "greedy_arms.txt"), "w") as f:
        for layer_idx, arm_idx in enumerate(greedy_arms):
            perm_name = flow.permutation_names[arm_idx]
            f.write(f"Layer {layer_idx}: Arm {arm_idx} -> {perm_name}\n")


def plot_bandit_values(flow, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    values = flow.bandit_values.detach().cpu().numpy()
    num_layers, num_arms = values.shape
    arm_labels = np.arange(num_arms)

    for layer_idx in range(num_layers):
        plt.figure(figsize=(16, 5))
        plt.bar(arm_labels, values[layer_idx])
        plt.xlabel("Bandit Arm Index")
        plt.ylabel("Estimated Reward")
        plt.title(f"Layer {layer_idx}: Estimated Bandit Values")
        plt.xticks(arm_labels)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"layer_{layer_idx}_values.png"))
        plt.close()


def train_one_epoch(
    flow,
    trainloader,
    optimizer,
    device,
    epsilon,
    data_bins,
    epoch,
    global_step,
    history,
):
    flow.train()

    epoch_nll_loss = 0.0
    num_batches = 0

    for batch_idx, (inputs, _) in enumerate(trainloader, start=1):
        inputs = dequantize(inputs, data_bins=data_bins)
        inputs = inputs.view(inputs.shape[0], -1).to(device)

        chosen_arms = flow.select_arm_indices(epsilon=epsilon)

        optimizer.zero_grad()

        log_prob = flow(inputs, arm_indices=chosen_arms)
        nll_loss = -log_prob.mean()

        nll_loss.backward()
        optimizer.step()

        reward = -float(nll_loss.item())
        flow.update_bandits(chosen_arms, reward)

        global_step += 1
        num_batches += 1

        epoch_nll_loss += nll_loss.item()

        history["selected_arms_per_batch"].append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "batch_idx": batch_idx,
                "arms": list(chosen_arms),
            }
        )

        if global_step % checkpoint_every_minibatches == 0:
            save_checkpoint(
                flow=flow,
                optimizer=optimizer,
                epoch=epoch,
                batch_idx=batch_idx,
                global_step=global_step,
                checkpoint_dir=checkpoint_dir,
                history=history,
            )

    return {
        "nll_loss": epoch_nll_loss / num_batches,
        "global_step": global_step,
    }


def evaluate(flow, dataloader, device, data_bins):
    flow.eval()

    epoch_nll_loss = 0.0
    num_batches = 0

    greedy_arms = flow.greedy_arm_indices()

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = dequantize(inputs, data_bins=data_bins)
            inputs = inputs.view(inputs.shape[0], -1).to(device)

            log_prob = flow(inputs, arm_indices=greedy_arms)
            nll_loss = -log_prob.mean()

            epoch_nll_loss += nll_loss.item()
            num_batches += 1

    return {
        "nll_loss": epoch_nll_loss / num_batches,
    }


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.sample_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.analysis_dir, exist_ok=True)

    sample_shape = [1, 28, 28]

    trainloader, testloader = get_mnist_dataloaders(
        batch_size=args.batch_size,
    )

    flow = nice.NICE(
        coupling=args.coupling,
        in_out_dim=28 * 28,
        mid_dim=args.mid_dim,
        hidden=args.hidden,
        device=device,
        image_size=28,
        num_permutations=args.num_permutations,
        fixed_mask_config=args.fixed_mask_config,
        data_bins=args.data_bins,
    ).to(device)

    optimizer = torch.optim.Adam(flow.parameters(), lr=args.lr)

    history = {
        "train_nll": [],
        "test_nll": [],
        "greedy_arms": [],
        "bandit_values": [],
        "bandit_counts": [],
        "selected_arms_per_batch": [],
    }

    save_permutation_metadata(
        flow,
        os.path.join(args.analysis_dir, "permutation_metadata.json"),
    )
    save_permutation_summary_text(
        flow,
        os.path.join(args.analysis_dir, "permutation_summary.txt"),
    )

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            flow=flow,
            trainloader=trainloader,
            optimizer=optimizer,
            device=device,
            epsilon=args.epsilon,
            data_bins=args.data_bins,
            epoch=epoch,
            global_step=global_step,
            history=history,
        )
        global_step = train_metrics["global_step"]

        test_metrics = evaluate(
            flow=flow,
            dataloader=testloader,
            device=device,
            data_bins=args.data_bins,
        )

        history["train_nll"].append(train_metrics["nll_loss"])
        history["test_nll"].append(test_metrics["nll_loss"])
        history["greedy_arms"].append(flow.greedy_arm_indices())
        history["bandit_values"].append(flow.bandit_values.detach().cpu().numpy().copy())
        history["bandit_counts"].append(flow.bandit_counts.detach().cpu().numpy().copy())

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train NLL: {train_metrics['nll_loss']:.4f} | "
            f"Test NLL: {test_metrics['nll_loss']:.4f} | "
            f"Greedy Arms: {flow.greedy_arm_indices()}"
        )

        save_samples(
            flow=flow,
            epoch=epoch,
            sample_shape=sample_shape,
            sample_dir=args.sample_dir,
            num_samples=args.sample_size,
        )

    final_ckpt = os.path.join(args.checkpoint_dir, "nice_final.pt")
    torch.save(
        {
            "epoch": args.epochs,
            "global_step": global_step,
            "model_state_dict": flow.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "greedy_arms": flow.greedy_arm_indices(),
            "permutation_metadata": flow.get_permutation_metadata(),
        },
        final_ckpt,
    )

    with open("training_history.pkl", "wb") as f:
        pickle.dump(history, f)

    epochs_axis = range(1, args.epochs + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_axis, history["train_nll"], label="Train NLL")
    plt.plot(epochs_axis, history["test_nll"], label="Test NLL")
    plt.xlabel("Epoch")
    plt.ylabel("NLL")
    plt.title("MNIST NICE with Per-Layer Bandit Permutations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_nll.png")
    plt.close()

    plot_bandit_histograms(flow, args.analysis_dir)
    plot_bandit_values(flow, args.analysis_dir)

    with open(os.path.join(args.analysis_dir, "final_greedy_arms.json"), "w") as f:
        json.dump(
            {
                "greedy_arms": flow.greedy_arm_indices(),
                "greedy_arm_names": [
                    flow.permutation_names[idx] for idx in flow.greedy_arm_indices()
                ],
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--sample_size", type=int, default=100)

    parser.add_argument("--coupling", type=int, default=4)
    parser.add_argument("--mid_dim", type=int, default=1000)
    parser.add_argument("--hidden", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--epsilon", type=float, default=0.6)
    parser.add_argument("--num_permutations", type=int, default=40)
    parser.add_argument("--fixed_mask_config", type=int, default=0)

    parser.add_argument("--checkpoint_every_epochs", type=int, default=1)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--sample_dir", type=str, default="samples")
    parser.add_argument("--analysis_dir", type=str, default="analysis")

    parser.add_argument("--data_bins", type=float, default=256.0)

    args = parser.parse_args()
    main(args)