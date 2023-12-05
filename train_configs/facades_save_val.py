from utils.configuration import Hyperparams, Dataparams, Generalparams, Configuration

configuration = Configuration(
        Hyperparams(
            epochs=1000,
            lr=2e-4,
            batch_size=1,

            adam_beta_1=0.5,
            adam_beta_2=0.999,
            weights_decay=0.0,

            gen_loss_weight=1.0,
            disc_loss_weight=1.0,
            l1_lambda=10.0,

            pretrained_weights='runs/checkpoints/Pix2Pix_batchSize1_facadesDataset_directionb_to_a_L1LossLambda10.0_eepoch=39_trainGenLosstrain_combined_gen_loss=1.02_valGenLossval_combined_gen_loss=0.87.ckpt', # None or path to pretrained_weights
            gen_input_size=256,      # Generator prepared for 256 input and output as in the paper
        ),

        Dataparams(
            dataset_name='facades', # facades, maps, edges2shoes edges2handbas, cityscapes, night2day
            prepare_data_per_node=True, # Download dataset if it is now available
            direction='b_to_a', # a_to_b or b_to_a
        ),

        Generalparams(
            num_workers=8,
            accelerator='cuda', # cuda, cpu
            devices=[0,], # if cuda: comma separated list with the devices to be used. if cpu: number of cores to use
            test_model_epoch=10,
        ),
        verbose=True)
