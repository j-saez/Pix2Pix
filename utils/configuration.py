from dataclasses import dataclass
import importlib

@dataclass
class Generalparams:
    test_model_epoch: int
    num_workers: int
    accelerator: str
    devices: list

    def __str__(self):
        output = 'General params: \n'
        output += f'Accelerator: {self.accelerator}\n'
        output += f'GPUs to be used: {self.devices}\n'
        output += f'Num workers: {self.num_workers}\n'
        output += f'Test model every n epochs: {self.test_model_epoch}\n'
        return output

@dataclass
class Dataparams:
    dataset_name: str
    prepare_data_per_node: bool
    direction: str

    def __str__(self):
        output = 'Dataset parameters: \n'
        output += f'Dataset name: {self.dataset_name}\n'
        output += f'Prepare data per node: {self.prepare_data_per_node}\n'
        output += f'Direction: {self.direction}\n'
        return output

@dataclass
class Hyperparams:
    epochs: int
    n_epochs_decay: int
    lr: float
    batch_size: int

    adam_beta_1: float
    adam_beta_2: float
    weights_decay: float

    gen_loss_weight: float
    disc_loss_weight: float
    l1_lambda: float

    pretrained_weights: str
    gen_input_size: int

    def __str__(self):
        output = 'Hyperparams:\n'
        output += f'\tEpochs: {self.epochs}\n' 
        output += f'\nn_epochs_decay: {self.n_epochs_decay}\n' 
        output += f'\tLearning rate: {self.lr}\n'
        output += f'\tBath size: {self.batch_size}\n' 

        output += f'\tAdam beta 1: {self.adam_beta_1}\n' 
        output += f'\tAdam beta 2: {self.adam_beta_2}\n' 
        output += f'\tWeights decay: {self.weights_decay}\n' 

        output += f'\tGenerator loss weight: {self.gen_loss_weight}\n' 
        output += f'\tDiscriminator loss weight: {self.disc_loss_weight}\n' 
        output += f'\tL1 lambda: {self.l1_lambda}\n' 

        output += f'\tPretrained weights: {self.pretrained_weights}\n' 
        output += f'\tGenerator input size: {self.gen_input_size}\n' 

        return output
    
class Configuration:
    def __init__(self, hyperparams: Hyperparams, dataparams: Dataparams, generalparams: Generalparams, verbose: bool=True) -> None:
        """
        Configuration class
        Inputs:
            >> hyperparams: (Hyperparams) Hyperparameters object
            >> dataparams: (Dataparams) Dataset params object
            >> generalparams: (Generalparams) General params object
        """
        self.hyperparams = hyperparams
        self.dataparams = dataparams
        self.general = generalparams

        if verbose:
            print(self.hyperparams)
            print(f'******')
            print(self.dataparams)
            print(f'******')
            print(self.general)
            print(f'******')

        return

def load_configuration(config_file_name: str):
    """
    Loads the configuration dict from the especified file.
    The configuration file needs to be stored in the train_configs directory.

    Inputs:
        >> config_file_name: (str) Name of the file containing the configuration.
    Outputs:
        >> configuration: (Configuration) Configuration dict for the training process.

    """
    config_path = os.path.join("train_configs", f"{config_file_name}.py")
    
    spec = importlib.util.spec_from_file_location(config_file_name, config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return module.configuration
