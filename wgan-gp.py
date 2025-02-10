import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import time
output_dir = r"C:\Users\lin37\Desktop\csds465\csds465HW3\image"
os.makedirs(output_dir, exist_ok=True)

# define resblcok
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, upsample=False):
        super(ResBlock, self).__init__()
        self.downsample = downsample
        self.upsample = upsample
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.relu = nn.ReLU()
    #handle down and up sample
    def forward(self, x):
        residual = x
        if self.upsample:  
            x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
            residual = nn.functional.interpolate(residual, scale_factor=2, mode='nearest')
            residual = self.skip(residual)
        elif self.downsample: 
            residual = self.skip(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.relu(x + residual)


# Generator change the resblock for different varaition
class Generator(nn.Module):
    def __init__(self, z_dim=128, img_channels=3):
        super(Generator, self).__init__()
        self.fc = nn.Linear(z_dim, 4 * 4 * 256)
        self.res1 = ResBlock(256, 256, upsample=True)
        self.res2 = ResBlock(256, 256, upsample=True)
        self.res3 = ResBlock(256, 256, upsample=True)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        z = self.fc(z).view(-1, 256, 4, 4)
        z = self.res1(z) 
        z = self.res2(z)  
        z = self.res3(z)  
        z= self.conv(z) 
        return z 


# discriminator
class Discriminator(nn.Module):
    def __init__(self, img_channels=3):
        super(Discriminator, self).__init__()
        self.res1 = ResBlock(img_channels, 128, downsample=True)
        self.res2 = ResBlock(128, 128, downsample=True)
        self.res3 = ResBlock(128, 128)
        self.res4 = ResBlock(128, 128)
        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.relu(x)
        x = self.global_pool(x).view(x.size(0), -1)
        return self.fc(x)

#gradient penalty
def gradient_penalty(critic, real, fake, device):
    batch_size, c, h, w = real.size()
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device, requires_grad=True)
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated = interpolated.to(device)
    interpolated.requires_grad_(True)
    mixed_scores = critic(interpolated)
    gradient = autograd.grad(
        inputs=interpolated,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores, device=device),
        create_graph=True,
        retain_graph=True
    )[0]
    gradient = gradient.view(gradient.size(0), -1)
    penalty = ((gradient.norm(2, dim=1) - 1) ** 2).mean()
    return penalty

if __name__ == "__main__":
    import time
    import os
    from torchvision.utils import save_image

    #training parameters
    z_dim = 128
    batch_size = 128
    lr = 2e-4
    lambda_gp = 10
    n_critic = 5
    total_iterations = 50000
    output_dir = "./generated_images5" 
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #data prprocess
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #modeles
    generator = Generator(z_dim=z_dim).to(device)
    critic = Discriminator().to(device)
    gen_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0, 0.9))
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr, betas=(0, 0.9))

    #training process
    loop = 0
    start_time = time.time()
    print("starting training...")

    for epoch in range(100):  
        for real_imgs, _ in dataloader:
            if loop >= total_iterations: 
                break
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            #train discriminator
            for _ in range(n_critic):
                z = torch.randn(batch_size, z_dim).to(device)
                fake_imgs = generator(z)
                real_score = critic(real_imgs)
                fake_score = critic(fake_imgs.detach())
                gp = gradient_penalty(critic, real_imgs, fake_imgs, device)
                critic_loss = -(real_score.mean() - fake_score.mean()) + lambda_gp * gp

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

            #train generator
            z = torch.randn(batch_size, z_dim).to(device)
            fake_imgs = generator(z)
            fake_score = critic(fake_imgs)
            generator_loss = -fake_score.mean()
            gen_optimizer.zero_grad()
            generator_loss.backward()
            gen_optimizer.step()
            loop += 1
            if loop % 1 == 0:
                elapsed_time = time.time() - start_time
                print(f"loop {loop}/{total_iterations} - discriminator Loss: {critic_loss.item():.4f}, "
                      f"generatorLoss: {generator_loss.item():.4f}, Time: {elapsed_time:.2f} seconds")

        if loop >= total_iterations:
            break

    print("generating images...")

    # generate 100 samples
    generator.eval()
    with torch.no_grad():
        z = torch.randn(100, z_dim).to(device)
        samples = generator(z)
        samples = (samples + 1) / 2 
        for i in range(100):
            save_path = os.path.join(output_dir, f"sample_{i + 1}.png")
            save_image(samples[i], save_path)

    print(f"save image to : {output_dir}")

   
