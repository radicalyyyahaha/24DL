import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Construct generator. You should experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity
        self.model = nn.Sequential(
            nn.Linear(args.latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()  
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You should experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()  
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    adversarial_loss = nn.BCELoss()  

    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            batch_size = imgs.size(0)
            imgs = imgs.to(device)

            valid = torch.ones(batch_size, 1, device=device, dtype=torch.float32)
            fake = torch.zeros(batch_size, 1, device=device, dtype=torch.float32)

            # -----------------
            # Train Generator
            # -----------------
            optimizer_G.zero_grad()

            z = torch.randn(batch_size, args.latent_dim, device=device)
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            real_loss = adversarial_loss(discriminator(imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            if i % 50 == 0:
                print(
                    f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(dataloader)}] "
                    f"[D loss: {d_loss.item()}] [G loss: {g_loss.item()}]"
                )

            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                save_image(gen_imgs[:25],
                           f'./Part 2/images/{batches_done}.png',
                           nrow=5, normalize=True, value_range=(-1, 1))
                
            if epoch == 0:
                torch.save(generator.state_dict(), "generator_start.pt")
            elif epoch == args.n_epochs // 2:
                torch.save(generator.state_dict(), "generator_halfway.pt")
            elif epoch == args.n_epochs - 1:
                torch.save(generator.state_dict(), "mnist_generator.pt")


def main():
    # Create output image directory
    os.makedirs('./Part 2/images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./Part 2/data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5),
                                                (0.5))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    main()
