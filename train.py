import os
import argparse
import torchvision
import pytorch_lightning as pl
from models.pix2pix                      import Pix2Pix
from utils.tools                         import load_configuration
from pytorch_lightning.loggers           import TensorBoardLogger
from datasets.classes.pix2pix_datamodule import Pix2PixDataModule

SOURCE_IMGS_IDX = 0
TARGET_IMGS_IDX = 1
CHS_IDX = 1

if __name__ == '__main__':

    #################
    # CONFIGURATION #
    #################

    parser = argparse.ArgumentParser(description='Arguments for pix2pix inference.')
    parser.add_argument( '--config-file',        type=str, required=True, help='Path to the configuration file.' )
    args = parser.parse_args()

    config = load_configuration(args.config_file)
    if config.hyperparams.gen_input_size != 256:
        raise ValueError(f'The gen just accepts an input size of 256x256. Received: {config.hyperparams.gen_input_size}')

    ##################
    # MODEL and DATA #
    ##################

    data_module = Pix2PixDataModule(config)
    fixed_data = data_module.get_fixed_data(total_imgs=4)

    std, mean  = [0.5] * fixed_data[0].shape[1], [0.5] * fixed_data[0].shape[1]
    denorm_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(
            [-m/s for m, s in zip(mean, std)], [1/s for s in std]
        ),
    ])


    model = pl.LightningModule()
    if config.hyperparams.pretrained_weights == None:
        model = Pix2Pix(
            # Src and tgt datasets must much the number or channels.
            source_imgs_chs=fixed_data[SOURCE_IMGS_IDX].shape[CHS_IDX],
            target_imgs_chs=fixed_data[TARGET_IMGS_IDX].shape[CHS_IDX],
            conf=config,
            fixed_data=fixed_data,
            denorm_transforms=denorm_transforms)
    else:
        print(f'Loading pretrained weights from: {config.hyperparams.pretrained_weights}')
        model = Pix2Pix.load_from_checkpoint(config.hyperparams.pretrained_weights)

    #############
    # CALLBACKS #
    #############

    checkpoint_filename = f"Pix2Pix_batchSize{config.hyperparams.batch_size}_{config.dataparams.dataset_name}Dataset_direction{config.dataparams.direction}_L1LossLambda{config.hyperparams.l1_lambda}"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f'{os.getcwd()}/runs/checkpoints',
        filename=checkpoint_filename+'_e{epoch}_trainGenLoss{train_combined_gen_loss:.2f}_valGenLoss{val_combined_gen_loss:.2f}',
        save_top_k = 1,
        monitor='val_combined_gen_loss',
        mode='min',
        verbose=False,
        save_on_train_epoch_end=True)

    tb_logger = TensorBoardLogger(
        save_dir='runs/tensorboard',
        name='Pix2Pix',
        version=f"batchSize{config.hyperparams.batch_size}_{config.dataparams.dataset_name}_{config.dataparams.direction}_UNETGen_Patch70Disc_{config.hyperparams.l1_lambda}L1LossLambda",
        default_hp_metric=False)

    #########
    # Train #
    #########

    trainer = pl.Trainer(
        callbacks = [checkpoint_callback],
        logger=tb_logger,
        accelerator=config.general.accelerator,
        devices=config.general.devices,
        max_epochs=config.hyperparams.epochs,
        log_every_n_steps=10)

    trainer.fit(model, data_module)
