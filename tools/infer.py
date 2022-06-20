import logging
import hydra
from omegaconf import DictConfig
import torch
from torchvision import transforms
from PIL import Image
from tools.train import initialize_model
import os


@hydra.main(version_base=None, config_path="../conf", config_name="infer")
def main(cfgs: DictConfig):
    """a simple model inference script for scene recognizing(places365 challenge).
        http://places2.csail.mit.edu/download.html
    """
    device = torch.device(cfgs.device) if torch.cuda.is_available() else torch.device('cpu')
    logger = logging.getLogger('model inference')
    logger.info("infer on device:{}".format(device))
    model, input_size = initialize_model(cfgs.arch, num_classes=365, feature_extract=False, use_pretrained=False)
    torch.save(model.state_dict(), os.path.join("/root/TorchScene/checkpoints/", cfgs.arch) + '.pt')
    logger.info("loading model weights from {}".format(cfgs.weight_path))
    model.load_state_dict(torch.load(cfgs.weight_path))
    logger.info("loading successfully.")
    model = model.to(device)
    model.eval()
    data_transform = transforms.Compose([
        transforms.Resize(
            input_size,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(cfgs.img_path)
    logger.info("input image is loaded from {}".format(cfgs.img_path))
    input = data_transform(img).unsqueeze(0).to(device)
    outputs = model(input).to('cpu')
    predict = int(torch.argmax(torch.softmax(outputs, dim=-1)))
    with open(cfgs.categories_map_dir, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if predict == int(line.split(' ')[-1]):
                logger.info("prediction of {} is {}".format(cfgs.img_path, line.split(' ')[0].split('/')[-1]))
                break


if __name__ == "__main__":
    main()





