import torch
from typing import List
import torch.nn.functional as F
import torchvision
import torch.nn          as nn
import pytorch_lightning as torch_lightning
from utils.configuration          import Configuration
from models import UNETGenerator, PatchGAN70Disc

SRC = 0
REAL_TGT = 1
FAKE_TGT = 2

GEN_OPT  = 0
DISC_OPT = 1

FIXED_TRAIN_SRC_IMGS_IDX = 0
FIXED_TRAIN_TGT_IMGS_IDX = 1
FIXED_TEST_SRC_IMGS_IDX  = 2
FIXED_TEST_TGT_IMGS_IDX  = 3

class Pix2Pix(torch_lightning.LightningModule):

    def __init__( self, source_imgs_chs: int, target_imgs_chs: int, conf: Configuration, fixed_data: list, denorm_transforms: torchvision.transforms.Compose) -> None:
        """
        Class of the Pix2Pix model described in the original paper.
        Inputs:
            >> source_imgs_chs: (int) Number of channels in the source images.
            >> target_imgs_chs: (int) Number of channels in the target images.
            >> conf: (Configuraiton)
            >> fixed_data: (torch.tensor) Fixed data to show the evolution of the generated images during training.
            >> denorm_transforms: (torchvision.transforms.Compose) Denormalization transforms 

        Attributes:
            >> conf:(Configuration) Configuration for the training process and some parameters of the model.
            >> generator: (nn.Module) Generator model.
            >> discriminator: (nn.Module) Discriminator model.
            >> adv_loss: (nn.BCEWithLogitsLoss)
            >> l1_loss: (nn.L1Loss)
        """
        super().__init__()
        self.conf = conf

        # Set to manual optimization as we are working with multiple optimizers at the same time
        self.save_hyperparameters()
        self.automatic_optimization = False

        print('===== Loading Generator model =====')
        self.generator = UNETGenerator(source_imgs_chs, target_imgs_chs)

        print('===== Loading Discrinator model =====')
        self.discriminator = PatchGAN70Disc(source_imgs_chs)

        self.adv_loss = torch.nn.BCEWithLogitsLoss() 
        self.l1_loss = nn.L1Loss()

        self.fixed_data = fixed_data 
        self.denorm_transforms = denorm_transforms 

        if not conf.hyperparams.pretrained_weights:
            self.initialise_weights()

        print(f'************ Model loaded ************\n\n\n')
        return

    def initialise_weights(self):
        """
        Weight initialization for the different layers of the model following the especifications that appear in the original paper.
        Inputs: None
        Outputs: None
        """
        print(f'Initialising weights as especified in the paper.')
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # Initialize weights for convolutional, transpose convolutional, and linear layers
                nn.init.normal_(m.weight, mean=0, std=0.02)
                if m.bias is not None:
                    # Initialize biases if they exist
                    nn.init.constant_(m.bias, 0)
        return

    def configure_optimizers(self,):
        """
        Configures the optimizer that will be used during the training process
        Inputs: None
        Outputs:
            >> optimizers_list: (list) Contains the optimizers for the generator, discriminator and classifier.
        """
        generator_optim = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.conf.hyperparams.lr,
            weight_decay=self.conf.hyperparams.weights_decay,
            betas=(self.conf.hyperparams.adam_beta_1, self.conf.hyperparams.adam_beta_2))

        discriminator_optim = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.conf.hyperparams.lr,
            weight_decay=self.conf.hyperparams.weights_decay,
            betas=(self.conf.hyperparams.adam_beta_1,
                   self.conf.hyperparams.adam_beta_2))

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - self.conf.hyperparams.epochs) / float(self.conf.hyperparams.n_epochs_decay + 1)
            return lr_l

        gen_lr_sched = torch.optim.lr_scheduler.LambdaLR(generator_optim, lr_lambda=lambda_rule)
        disc_lr_sched = torch.optim.lr_scheduler.LambdaLR(discriminator_optim, lr_lambda=lambda_rule)

        optimizers_list = [generator_optim, discriminator_optim]
        lr_sched_list = [gen_lr_sched, disc_lr_sched]
        return optimizers_list, lr_sched_list

    def training_step(self, batch: list, batch_idx: int):
        """
        Peforms a one training step for the Generator and Discriminator.
        Inputs:
            >> batch: (list of torch.tensors) It contains the [source_dom_img, targe_dom_img].
            >> batch_idx: (int) Index of the current batch.
        Outputs:
            >> loss_dict: (dict) Dict of loss containing 'gen_loss', 'disc_loss' and 'task_loss'.
        """
        self.__common_step__(batch, 'train', train=True)
        return

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int):
        """
        Peforms a one validation step for the Generator, Discriminator.
        Inputs:
            >> batch: (list of torch.tensors) It contains the [source_dom_img, targe_dom_img].
            >> batch_idx: (int) Index of the current batch.
        Outputs: None
        """
        self.__common_step__(batch, 'val', train=False)
        return

    def __common_step__(self, batch: List[torch.Tensor], stage: str, train: bool=True):
        """
        Peforms a common step for the training, validation and test data splits.
        """
        with torch.cuda.amp.autocast():
            batch.append(self.generator(batch[SRC]))
            combined_disc_loss, disc_real_loss, disc_fake_loss  = self.train_discriminator(batch, train)
            combined_gen_loss, bce_loss_gen, l1_loss_gen= self.train_generator(batch, train)

            loss_dict = {
                f'{stage}_comb_disc_loss': combined_disc_loss,
                f'{stage}_disc_real_loss': disc_real_loss,
                f'{stage}_disc_fake_loss': disc_fake_loss,
                f'{stage}_combined_gen_loss': combined_gen_loss,
                f'{stage}_bce_loss_gen': bce_loss_gen,
                f'{stage}_l1_loss_gen': l1_loss_gen*self.conf.hyperparams.l1_lambda,
            }

            self.log_dict(loss_dict, on_step=False, on_epoch=True)
            return

    def on_train_epoch_end(self):

        self.lr_schedulers()[DISC_OPT].step()

        if self.current_epoch % self.conf.general.test_model_epoch == 0:
            source_imgs = self.fixed_data[FIXED_TRAIN_SRC_IMGS_IDX].to(self.device)
            target_imgs = self.fixed_data[FIXED_TRAIN_TGT_IMGS_IDX].to(self.device)

            combined_images = torch.cat(
                dim=2,
                tensors=(
                    self.denorm_transforms(source_imgs),
                    self.denorm_transforms(target_imgs),
                    self.denorm_transforms(self.generator(source_imgs))))

            combined_images_grid = torchvision.utils.make_grid(combined_images)
            self.logger.experiment.add_image('Train images', combined_images_grid, self.current_epoch)

        return

    def on_validation_epoch_end(self) -> None:
        if self.current_epoch % self.conf.general.test_model_epoch == 0:
            source_imgs = self.fixed_data[FIXED_TEST_SRC_IMGS_IDX].to(self.device)
            target_imgs = self.fixed_data[FIXED_TEST_TGT_IMGS_IDX].to(self.device)

            combined_images = torch.cat(
                dim=2,
                tensors=(
                    self.denorm_transforms(source_imgs),
                    self.denorm_transforms(target_imgs),
                    self.denorm_transforms(self.generator(source_imgs))))

            combined_images_grid = torchvision.utils.make_grid(combined_images)
            self.logger.experiment.add_image('validation images', combined_images_grid, self.current_epoch)
        return

    def train_discriminator(self, images: List[torch.Tensor], train: bool):
        """
        Trains the discriminator one epoch: max log(D(x)) + log(1-D(G(z)))
        Inputs:
            >> TODO
        Outputs:
            >> loss_disc: (torch.float32) containing the value of the loss for the discriminator
        """
        real_disc_output = self.discriminator(images[SRC], images[REAL_TGT].detach())
        fake_disc_output = self.discriminator(images[SRC], images[FAKE_TGT].detach())

        loss_real_disc = self.adv_loss(real_disc_output, torch.ones_like(real_disc_output))
        loss_fake_disc = self.adv_loss(fake_disc_output, torch.zeros_like(fake_disc_output))
        disc_lambda = self.conf.hyperparams.disc_loss_weight
        combined_loss =  disc_lambda * (loss_real_disc + loss_fake_disc) / 2.0

        if train:
            self.optimizers()[DISC_OPT].zero_grad()
            combined_loss.backward()
            self.optimizers()[DISC_OPT].step()
        return combined_loss, loss_real_disc, loss_fake_disc

    def train_generator(self, images: List[torch.Tensor], train: bool):
        """
        Trains the generator one epoch: min log(1-D(G(z))) <--> max log(D(G(z)))
        Inputs: TODO
        Outputs:
            >> loss_generator: (torch.float32) containing the value of the loss for the generator.
        """
        gen_lambda = self.conf.hyperparams.gen_loss_weight
        l1_lambda  =  self.conf.hyperparams.l1_lambda

        fake_disc_output = self.discriminator(images[SRC], images[FAKE_TGT]).reshape(-1)
        bce_loss_generator = self.adv_loss(fake_disc_output, torch.ones_like(fake_disc_output))
        l1_loss_generator = self.l1_loss(images[FAKE_TGT], images[REAL_TGT])
        combined_loss = gen_lambda * bce_loss_generator + l1_loss_generator * l1_lambda 

        if train:
            self.optimizers()[GEN_OPT].zero_grad()
            combined_loss.backward()
            self.optimizers()[GEN_OPT].step()

        return combined_loss, bce_loss_generator, l1_loss_generator


    def forward(self, source_dom_img: torch.Tensor):
        """
        Predicts a generated image.
        Inputs:
            >> source_dom_img: (torch.Tensor shape=[1,chs,h,w])
        Outputs:
            >> generated_img: (torch.Tensor shape=[1,chs,h,w])
        """
        return self.generator(source_dom_img)
