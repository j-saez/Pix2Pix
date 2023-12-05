import torch
import torchvision
import argparse
from PIL import Image
from utils.tools    import load_configuration
from models.pix2pix import Pix2Pix
from datasets.classes.pix2pix_dataset import Pix2PixDataset
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for pix2pix inference.')
    parser.add_argument( '--config-file',        type=str, required=True, help='Path to the configuration file.' )
    args = parser.parse_args()

    config = load_configuration(args.config_file)
    if config.hyperparams.gen_input_size != 256:
        raise ValueError(f'The gen just accepts an input size of 256x256. Received: {config.hyperparams.gen_input_size}')

    if config.hyperparams.pretrained_weights == None:
        raise ValueError(f'Pretrained weights must be especified to .lot all the validation results.')

    print(f'Loading pretrained weights from: {config.hyperparams.pretrained_weights}')
    model = Pix2Pix.load_from_checkpoint(config.hyperparams.pretrained_weights)

    dataset = Pix2PixDataset(config, 'val')

    std, mean  = [0.5] * dataset[0][0].shape[0], [0.5] * dataset[0][0].shape[0]
    denorm_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(
            [-m/s for m, s in zip(mean, std)], [1/s for s in std]
        ),
    ])

    with torch.no_grad():
        for i in range(len(dataset)):

            device = model.device
            src_img = dataset[i][0].to(device)
            tgt_img = dataset[i][1].to(device)
            fake_tgt = model(src_img.unsqueeze(dim=0)).squeeze(dim=0)

            combined_images = torch.cat(
                dim=2,
                tensors=(
                    denorm_transforms(src_img),
                    denorm_transforms(tgt_img),
                    denorm_transforms(fake_tgt)))

            combined_images_grid = torchvision.utils.make_grid(combined_images).permute(1,2,0).cpu().numpy()
            image_pil = Image.fromarray((combined_images_grid * 255).astype('uint8'))

            plt.imshow(image_pil)
            plt.axis('off')
            plt.show()
