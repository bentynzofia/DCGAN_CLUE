from src.models.gan import weights_init, Generator, Discriminator

netG, netD = Generator(), Discriminator()

netG.apply(weights_init)
netD.apply(weights_init)

