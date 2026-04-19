import argparse
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import nice


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset
# ============================================================

def get_digits_dataloaders(batch_size=128, test_size=0.2, seed=42):
    digits = load_digits()
    X = digits.images.astype(np.float32) / 16.0
    y = digits.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    trainset = TensorDataset(X_train, y_train)
    testset = TensorDataset(X_test, y_test)

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


# ============================================================
# Bandit
# ============================================================

class EpsilonGreedyBandit:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms, dtype=np.int64)
        self.values = np.zeros(n_arms, dtype=np.float64)

    def select_arm(self):
        untried = np.where(self.counts == 0)[0]
        if not (0.0 <= self.epsilon <= 1.0):
            raise ValueError("epsilon must be in [0, 1]")
        
        
        if len(untried) > 0:
            return int(untried[0])

        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.n_arms))

        return int(np.argmax(self.values))

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        old_value = self.values[arm]
        self.values[arm] = old_value + (reward - old_value) / n

    def best_arm(self):
        if np.all(self.counts == 0):
            return 0
        return int(np.argmax(self.values))


# ============================================================
# Mask bank helpers
# ============================================================

def make_grid_indices():
    rows = np.arange(8).reshape(8, 1)
    cols = np.arange(8).reshape(1, 8)
    rows, cols = np.broadcast_arrays(rows, cols)
    return rows.astype(np.float32), cols.astype(np.float32)


def top32_mask_from_scores(scores):
    flat = scores.reshape(-1)
    order = np.argsort(-flat, kind="mergesort")
    chosen = order[:32]

    mask = np.zeros(64, dtype=np.float32)
    mask[chosen] = 1.0
    return mask


def make_unique_mask(mask, scores, used_keys, arm_idx):
    """
    If a mask duplicates a previous arm, repair it deterministically
    by swapping selected/unselected entries while preserving exactly 32 ones.

    Preference:
    - preserve the score-family as much as possible
    - swap out the weakest selected cells
    - swap in the strongest unselected cells
    - use arm_idx-dependent cyclic ordering for deterministic diversity
    """
    key = tuple(mask.tolist())
    if key not in used_keys:
        return mask

    flat_scores = scores.reshape(-1)

    ones = np.where(mask == 1.0)[0]
    zeros = np.where(mask == 0.0)[0]

    # weakest selected first
    ones_sorted = ones[np.argsort(flat_scores[ones], kind="mergesort")]
    # strongest unselected first
    zeros_sorted = zeros[np.argsort(-flat_scores[zeros], kind="mergesort")]

    # deterministic cyclic shift depending on arm index
    if len(ones_sorted) > 0:
        shift_ones = arm_idx % len(ones_sorted)
        ones_sorted = np.roll(ones_sorted, -shift_ones)

    if len(zeros_sorted) > 0:
        shift_zeros = (3 * arm_idx) % len(zeros_sorted)
        zeros_sorted = np.roll(zeros_sorted, -shift_zeros)

    # try single swaps first
    for out_idx in ones_sorted:
        for in_idx in zeros_sorted:
            new_mask = mask.copy()
            new_mask[out_idx] = 0.0
            new_mask[in_idx] = 1.0
            new_key = tuple(new_mask.tolist())
            if new_key not in used_keys:
                return new_mask

    # then try 2 swaps if needed
    max_ones_try = min(12, len(ones_sorted))
    max_zeros_try = min(12, len(zeros_sorted))
    for i in range(max_ones_try):
        for j in range(i + 1, max_ones_try):
            out1, out2 = ones_sorted[i], ones_sorted[j]
            for p in range(max_zeros_try):
                for q in range(p + 1, max_zeros_try):
                    in1, in2 = zeros_sorted[p], zeros_sorted[q]
                    new_mask = mask.copy()
                    new_mask[out1] = 0.0
                    new_mask[out2] = 0.0
                    new_mask[in1] = 1.0
                    new_mask[in2] = 1.0
                    new_key = tuple(new_mask.tolist())
                    if new_key not in used_keys:
                        return new_mask

    raise ValueError(
        f"Could not construct a unique 32-entry mask for arm {arm_idx}."
    )


def row_family_scores():
    rows, cols = make_grid_indices()
    family = []

    row_weights_list = [
        np.array([8, 8, 8, 8, 1, 1, 1, 1], dtype=np.float32),
        np.array([1, 1, 1, 1, 8, 8, 8, 8], dtype=np.float32),
        np.array([8, 1, 8, 1, 8, 1, 8, 1], dtype=np.float32),
        np.array([1, 8, 1, 8, 1, 8, 1, 8], dtype=np.float32),
        np.array([8, 8, 1, 1, 1, 1, 8, 8], dtype=np.float32),
        np.array([1, 1, 8, 8, 8, 8, 1, 1], dtype=np.float32),
        np.array([8, 8, 1, 1, 8, 8, 1, 1], dtype=np.float32),
        np.array([1, 1, 8, 8, 1, 1, 8, 8], dtype=np.float32),
    ]

    col_biases = [
        0.01 * cols,
        -0.01 * cols,
        0.02 * ((cols % 2) == 0).astype(np.float32),
        0.02 * ((cols % 2) == 1).astype(np.float32),
        0.01 * np.abs(cols - 3.5),
        -0.01 * np.abs(cols - 3.5),
        0.01 * (cols < 4).astype(np.float32),
        0.01 * (cols >= 4).astype(np.float32),
    ]

    row_int = rows.astype(np.int64)
    for i in range(8):
        scores = row_weights_list[i][row_int] + col_biases[i]
        family.append(scores.astype(np.float32))

    return family


def col_family_scores():
    rows, cols = make_grid_indices()
    family = []

    col_weights_list = [
        np.array([8, 8, 8, 8, 1, 1, 1, 1], dtype=np.float32),
        np.array([1, 1, 1, 1, 8, 8, 8, 8], dtype=np.float32),
        np.array([8, 1, 8, 1, 8, 1, 8, 1], dtype=np.float32),
        np.array([1, 8, 1, 8, 1, 8, 1, 8], dtype=np.float32),
        np.array([8, 8, 1, 1, 1, 1, 8, 8], dtype=np.float32),
        np.array([1, 1, 8, 8, 8, 8, 1, 1], dtype=np.float32),
        np.array([8, 8, 1, 1, 8, 8, 1, 1], dtype=np.float32),
        np.array([1, 1, 8, 8, 1, 1, 8, 8], dtype=np.float32),
    ]

    row_biases = [
        0.01 * rows,
        -0.01 * rows,
        0.02 * ((rows % 2) == 0).astype(np.float32),
        0.02 * ((rows % 2) == 1).astype(np.float32),
        0.01 * np.abs(rows - 3.5),
        -0.01 * np.abs(rows - 3.5),
        0.01 * (rows < 4).astype(np.float32),
        0.01 * (rows >= 4).astype(np.float32),
    ]

    col_int = cols.astype(np.int64)
    for i in range(8):
        scores = col_weights_list[i][col_int] + row_biases[i]
        family.append(scores.astype(np.float32))

    return family


def checker_family_scores():
    rows, cols = make_grid_indices()
    family = []

    parity = ((rows + cols) % 2).astype(np.float32)
    block2 = (((rows // 2) + (cols // 2)) % 2).astype(np.float32)
    row_par = (rows % 2).astype(np.float32)
    col_par = (cols % 2).astype(np.float32)

    family.append(5.0 * (1.0 - parity) + 0.2 * rows + 0.01 * cols)
    family.append(5.0 * parity + 0.2 * rows + 0.01 * cols)
    family.append(5.0 * (1.0 - block2) + 0.2 * cols + 0.01 * rows)
    family.append(5.0 * block2 + 0.2 * cols + 0.01 * rows)
    family.append(4.0 * (1.0 - parity) + 2.0 * row_par + 0.1 * cols)
    family.append(4.0 * parity + 2.0 * col_par + 0.1 * rows)

    return [s.astype(np.float32) for s in family]


def border_center_family_scores():
    rows, cols = make_grid_indices()
    family = []

    dist_border = np.minimum(
        np.minimum(rows, cols),
        np.minimum(7.0 - rows, 7.0 - cols)
    ).astype(np.float32)

    dist_center = ((rows - 3.5) ** 2 + (cols - 3.5) ** 2).astype(np.float32)
    parity = ((rows + cols) % 2).astype(np.float32)

    family.append(-3.0 * dist_border - 0.05 * dist_center)
    family.append(3.0 * dist_border - 0.05 * dist_center)
    family.append(-2.0 * dist_border + 0.6 * parity)
    family.append(2.0 * dist_border + 0.6 * parity)
    family.append(-1.5 * dist_border + 0.8 * np.abs(cols - 3.5))
    family.append(-1.5 * dist_border + 0.8 * np.abs(rows - 3.5))

    return [s.astype(np.float32) for s in family]


def diagonal_family_scores():
    rows, cols = make_grid_indices()
    family = []

    d_main = np.abs(rows - cols).astype(np.float32)
    d_anti = np.abs(rows + cols - 7.0).astype(np.float32)

    family.append(-2.0 * d_main - 0.15 * d_anti)
    family.append(-2.0 * d_anti - 0.15 * d_main)
    family.append(-1.5 * np.minimum(d_main, d_anti) + 0.1 * rows)
    family.append(-1.5 * d_main + 0.8 * (cols < 4).astype(np.float32))
    family.append(-1.5 * d_anti + 0.8 * (rows < 4).astype(np.float32))
    family.append(
        -1.0 * d_main
        - 1.0 * d_anti
        + 0.2 * ((rows + cols) % 2).astype(np.float32)
    )

    return [s.astype(np.float32) for s in family]


def quadrant_family_scores():
    rows, cols = make_grid_indices()
    family = []

    q_tl = ((rows < 4) & (cols < 4)).astype(np.float32)
    q_tr = ((rows < 4) & (cols >= 4)).astype(np.float32)
    q_bl = ((rows >= 4) & (cols < 4)).astype(np.float32)
    q_br = ((rows >= 4) & (cols >= 4)).astype(np.float32)

    family.append(4.0 * q_tl + 4.0 * q_br + 0.1 * rows)
    family.append(4.0 * q_tr + 4.0 * q_bl + 0.1 * cols)
    family.append(
        4.0 * (cols < 4).astype(np.float32)
        + 1.0 * (np.abs(rows - cols) <= 1).astype(np.float32)
    )
    family.append(
        4.0 * (cols >= 4).astype(np.float32)
        + 1.0 * (np.abs(rows + cols - 7.0) <= 1).astype(np.float32)
    )
    family.append(
        4.0 * (rows < 4).astype(np.float32)
        + 1.5 * (((rows + cols) % 2) == 0).astype(np.float32)
    )
    family.append(
        4.0 * (rows >= 4).astype(np.float32)
        + 1.5 * (((rows + cols) % 2) == 1).astype(np.float32)
    )

    return [s.astype(np.float32) for s in family]


def build_mask_bank():
    score_families = []
    score_families.extend(row_family_scores())            # 8
    score_families.extend(col_family_scores())            # 8
    score_families.extend(checker_family_scores())        # 6
    score_families.extend(border_center_family_scores())  # 6
    score_families.extend(diagonal_family_scores())       # 6
    score_families.extend(quadrant_family_scores())       # 6

    if len(score_families) != 40:
        raise ValueError(f"Expected 40 score patterns, got {len(score_families)}")

    masks = []
    mask_keys = set()

    for i, scores in enumerate(score_families):
        mask = top32_mask_from_scores(scores)
        mask = make_unique_mask(mask, scores, mask_keys, i)

        if int(mask.sum()) != 32:
            raise ValueError(f"Arm {i} does not have exactly 32 active entries.")

        key = tuple(mask.tolist())
        if key in mask_keys:
            raise ValueError(f"Duplicate mask still present at arm {i}.")

        mask_keys.add(key)
        masks.append(torch.tensor(mask, dtype=torch.float32))

    return masks


# ============================================================
# Training / Testing
# ============================================================

def train_one_epoch(
    flow,
    trainloader,
    optimizer,
    device,
    bandit,
    mask_bank,
    arm_update_freq=10,
):
    flow.train()

    epoch_nll_sum = 0.0
    epoch_num_batches = 0

    current_arm = None
    block_nll_sum = 0.0
    block_batch_count = 0

    for batch_idx, (inputs, _) in enumerate(trainloader):
        if batch_idx % arm_update_freq == 0:
            if current_arm is not None and block_batch_count > 0:
                avg_block_nll = block_nll_sum / block_batch_count
                reward = -avg_block_nll
                bandit.update(current_arm, reward)

            current_arm = bandit.select_arm()
            flow.set_mask(mask_bank[current_arm].to(device))

            block_nll_sum = 0.0
            block_batch_count = 0

        inputs = inputs.view(inputs.shape[0], -1).to(device)

        optimizer.zero_grad()
        log_prob = flow(inputs)
        nll_loss = -log_prob.mean()
        nll_loss.backward()
        optimizer.step()

        nll_value = nll_loss.item()
        epoch_nll_sum += nll_value
        epoch_num_batches += 1

        block_nll_sum += nll_value
        block_batch_count += 1

    if current_arm is not None and block_batch_count > 0:
        avg_block_nll = block_nll_sum / block_batch_count
        reward = -avg_block_nll
        bandit.update(current_arm, reward)

    epoch_avg_nll = epoch_nll_sum / max(epoch_num_batches, 1)

    return {
        "nll": epoch_avg_nll,
        "best_arm": bandit.best_arm(),
    }


def test_one_epoch(flow, testloader, device, mask_bank, arm_idx):
    flow.eval()
    flow.set_mask(mask_bank[arm_idx].to(device))

    epoch_nll_sum = 0.0
    epoch_num_batches = 0

    with torch.no_grad():
        for inputs, _ in testloader:
            inputs = inputs.view(inputs.shape[0], -1).to(device)
            log_prob = flow(inputs)
            nll_loss = -log_prob.mean()

            epoch_nll_sum += nll_loss.item()
            epoch_num_batches += 1

    return epoch_nll_sum / max(epoch_num_batches, 1)


# ============================================================
# Visualization / Saving
# ============================================================

def save_samples(flow, epoch, device, output_dir, arm_idx, mask_bank, num_samples=100):
    flow.eval()
    flow.set_mask(mask_bank[arm_idx].to(device))

    with torch.no_grad():
        samples = flow.sample(num_samples).cpu().numpy()

    samples = samples.reshape(-1, 8, 8)

    s_min = samples.min()
    s_max = samples.max()
    samples = (samples - s_min) / (s_max - s_min + 1e-10)

    ncols = 10
    nrows = int(np.ceil(num_samples / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    idx = 0
    for r in range(nrows):
        for c in range(ncols):
            axes[r, c].axis("off")
            if idx < num_samples:
                axes[r, c].imshow(samples[idx], cmap="gray", vmin=0.0, vmax=1.0)
            idx += 1

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"epoch_{epoch:04d}.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def save_mask_visualizations(mask_bank, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    n_masks = len(mask_bank)
    ncols = 5
    nrows = int(np.ceil(n_masks / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 2 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    for idx in range(nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")
        if idx < n_masks:
            mask = mask_bank[idx].numpy().reshape(8, 8)
            axes[r, c].imshow(mask, cmap="gray", vmin=0.0, vmax=1.0)
            axes[r, c].set_title(f"Arm {idx}", fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "mask_bank.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main(args):
    set_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs("samples", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("mask_visualizations", exist_ok=True)

    trainloader, testloader = get_digits_dataloaders(
        batch_size=args.batch_size,
        test_size=0.2,
        seed=args.seed,
    )

    mask_bank = build_mask_bank()
    save_mask_visualizations(mask_bank, "mask_visualizations")

    full_dim = 8 * 8

    flow = nice.NICE(
        coupling=args.coupling,
        in_out_dim=full_dim,
        mid_dim=args.mid_dim,
        hidden=args.hidden,
        device=device,
        init_mask=mask_bank[0],
    ).to(device)

    optimizer = torch.optim.Adam(flow.parameters(), lr=args.lr)
    bandit = EpsilonGreedyBandit(n_arms=len(mask_bank), epsilon=args.epsilon)

    train_nll_history = []
    test_nll_history = []
    best_arm_history = []
    bandit_values_history = []
    bandit_counts_history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            flow=flow,
            trainloader=trainloader,
            optimizer=optimizer,
            device=device,
            bandit=bandit,
            mask_bank=mask_bank,
            arm_update_freq=args.arm_update_freq,
        )

        best_arm = bandit.best_arm()
        test_nll = test_one_epoch(
            flow=flow,
            testloader=testloader,
            device=device,
            mask_bank=mask_bank,
            arm_idx=best_arm,
        )

        train_nll_history.append(train_metrics["nll"])
        test_nll_history.append(test_nll)
        best_arm_history.append(best_arm)
        bandit_values_history.append(bandit.values.copy())
        bandit_counts_history.append(bandit.counts.copy())

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train NLL: {train_metrics['nll']:.4f} | "
            f"Test NLL (best arm {best_arm}): {test_nll:.4f} | "
            f"Bandit epsilon: {bandit.epsilon:.4f}"
        )

        if epoch % args.sample_every == 0:
            save_samples(
                flow=flow,
                epoch=epoch,
                device=device,
                output_dir="samples",
                arm_idx=best_arm,
                mask_bank=mask_bank,
                num_samples=args.sample_size,
            )

        if epoch % args.checkpoint_every == 0:
            ckpt_path = os.path.join("checkpoints", f"nice_epoch_{epoch:04d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": flow.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "bandit_counts": bandit.counts,
                    "bandit_values": bandit.values,
                    "best_arm": best_arm,
                },
                ckpt_path,
            )

    final_ckpt = os.path.join("checkpoints", "nice_final.pt")
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": flow.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "bandit_counts": bandit.counts,
            "bandit_values": bandit.values,
            "best_arm": bandit.best_arm(),
        },
        final_ckpt,
    )

    history = {
        "train_nll": train_nll_history,
        "test_nll": test_nll_history,
        "best_arm": best_arm_history,
        "bandit_values": bandit_values_history,
        "bandit_counts": bandit_counts_history,
    }

    with open("training_history.pkl", "wb") as f:
        pickle.dump(history, f)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.epochs + 1), train_nll_history, label="Train NLL")
    plt.plot(range(1, args.epochs + 1), test_nll_history, label="Test NLL")
    plt.xlabel("Epoch")
    plt.ylabel("NLL")
    plt.title("NICE on sklearn digits with bandit-selected direct masks")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_nll.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.epochs + 1), best_arm_history)
    plt.xlabel("Epoch")
    plt.ylabel("Best Arm Index")
    plt.title("Best Bandit Arm by Epoch")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("best_arm_history.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(mask_bank)), bandit.counts)
    plt.xlabel("Arm Index")
    plt.ylabel("Selections")
    plt.title("Final Bandit Arm Selection Counts")
    plt.tight_layout()
    plt.savefig("bandit_arm_counts.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--sample_every", type=int, default=10)
    parser.add_argument("--checkpoint_every", type=int, default=25)

    parser.add_argument("--coupling", type=int, default=4)
    parser.add_argument("--mid_dim", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--epsilon", type=float, default=0.8)
    parser.add_argument("--arm_update_freq", type=int, default=10)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)