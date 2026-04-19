"""Train Real NVP on MNIST"""
import argparse
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import util
import matplotlib.pyplot as plt
from models import RealNVP, RealNVPLoss
from tqdm import tqdm

def main(args):
    global best_loss
    device = 'cuda' if torch.cuda.is_available() and len(args.gpu_ids) > 0 else 'cpu'
    start_epoch = 0

    transform_train = transforms.Compose([
        transforms.Pad(2),
        #transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.Pad(2),
        #transforms.ToTensor()
    ])

    # Path where you manually saved your .npy file
    data_path = 'data/quickdraw_data.npy'
    
    # Use args.batch_size (matching your parser)
    trainset = QuickDrawDataset(npy_file=data_path, train=True, transform=transform_train, max_samples=20000)
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    testset = QuickDrawDataset(npy_file=data_path, train=False, transform=transform_test, max_samples=20000)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print('Building model..')
    net = RealNVP(num_scales=2, in_channels=1, mid_channels=64, num_blocks=8)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net, args.gpu_ids)
        cudnn.benchmark = args.benchmark

    if args.resume:
        # Load checkpoint.
        print('Resuming from checkpoint at ckpts/best.pth.tar...')
        assert os.path.isdir('ckpts'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('ckpts/best.pth.tar')
        net.load_state_dict(checkpoint['net'])
        global best_loss
        best_loss = checkpoint['test_loss']
        start_epoch = checkpoint['epoch']

    loss_fn = RealNVPLoss()
    param_groups = util.get_param_groups(net, args.weight_decay, norm_suffix='weight_g')
    optimizer = optim.Adam(param_groups, lr=args.lr)

    train_losses = []
    test_losses = []

    # FIXED: Single unified training loop
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        # Capture the average loss from the train/test functions
        avg_train_loss = train(epoch, net, trainloader, device, optimizer, loss_fn, args.max_grad_norm)
        avg_test_loss = test(epoch, net, testloader, device, loss_fn, args.num_samples)
        
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        # Plot every 5 epochs to see progress
        if (epoch + 1) % 5 == 0:
            plot_history(train_losses, test_losses)

    # Final plot
    plot_history(train_losses, test_losses, save_path='loss_plot.png')
import numpy as np
from torch.utils.data import Dataset

class QuickDrawDataset(Dataset):
    def __init__(self, npy_file, train=True, transform=None, max_samples=20000):
        # 1. Load the raw data
        full_data = np.load(npy_file)
        
        # 2. Truncate to exactly 20,000 images total
        if len(full_data) > max_samples:
            full_data = full_data[:max_samples]
        
        # 3. Perform 90/10 Split on the truncated data
        split = int(0.9 * len(full_data))
        if train:
            self.data = full_data[:split]
        else:
            self.data = full_data[split:]
            
        self.transform = transform
        print(f"{'Train' if train else 'Test'} set loaded with {len(self.data)} images.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # QuickDraw data is flat (784). Reshape to 28x28.
        img = self.data[idx].reshape(28, 28).astype(np.float32)
        
        # Scale to [0, 1]
        img /= 255.0
        
        # Convert to tensor and add channel dim: (1, 28, 28)
        img = torch.from_numpy(img).unsqueeze(0)
        
        if self.transform:
            img = self.transform(img)
            
        return img, 0
def plot_history(train_losses, test_losses, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue', linestyle='-')
    plt.plot(test_losses, label='Test Loss', color='red', linestyle='--')
    
    plt.title('RealNVP Training Progress (MNIST)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Negative Log-Likelihood)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()
    plt.close()

def test(epoch, net, testloader, device, loss_fn, num_samples):
    global best_loss
    net.eval()
    loss_meter = util.AverageMeter()
    with torch.no_grad():
        with tqdm(total=len(testloader.dataset)) as progress_bar:
            for x, _ in testloader:
                x = x.to(device)
                z, sldj = net(x, reverse=False)
                loss = loss_fn(z, sldj)
                loss_meter.update(loss.item(), x.size(0))
                progress_bar.set_postfix(loss=loss_meter.avg,
                                         bpd=util.bits_per_dim(x, loss_meter.avg))
                progress_bar.update(x.size(0))

    # Save checkpoint
    if loss_meter.avg < best_loss:
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'test_loss': loss_meter.avg,
            'epoch': epoch,
        }
        os.makedirs('ckpts', exist_ok=True)
        torch.save(state, 'ckpts/best.pth.tar')
        best_loss = loss_meter.avg

    return loss_meter.avg

def train(epoch, net, trainloader, device, optimizer, loss_fn, max_grad_norm):
    print('\nEpoch: %d' % epoch)
    net.train()
    loss_meter = util.AverageMeter()
    with tqdm(total=len(trainloader.dataset)) as progress_bar:
        for x, _ in trainloader:
            x = x.to(device)
            optimizer.zero_grad()
            z, sldj = net(x, reverse=False)
            loss = loss_fn(z, sldj)
            loss_meter.update(loss.item(), x.size(0))
            loss.backward()
            util.clip_grad_norm(optimizer, max_grad_norm)
            optimizer.step()

            progress_bar.set_postfix(loss=loss_meter.avg,
                                     bpd=util.bits_per_dim(x, loss_meter.avg))
            progress_bar.update(x.size(0))
    
    return loss_meter.avg

def sample(net, batch_size, device):
    """Sample from RealNVP model.

    Args:
        net (torch.nn.DataParallel): The RealNVP model wrapped in DataParallel.
        batch_size (int): Number of samples to generate.
        device (torch.device): Device to use.
    """
    z = torch.randn((batch_size, 1, 32, 32), dtype=torch.float32, device=device)
    x, _ = net(z, reverse=True)
    x = torch.sigmoid(x)

    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RealNVP on CIFAR-10')

    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--benchmark', action='store_true', help='Turn on CUDNN benchmarking')
    parser.add_argument('--gpu_ids', default='[0]', type=eval, help='IDs of GPUs to use')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=100., help='Max gradient norm for clipping')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs to train')
    parser.add_argument('--num_samples', default=64, type=int, help='Number of samples at test time')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of data loader threads')
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--weight_decay', default=5e-5, type=float,
                        help='L2 regularization (only applied to the weight norm scale factors)')

    best_loss = float('inf')

    main(parser.parse_args())
