from PIL import Image
from torch import Tensor
from money_counter import data, models, training
from torchvision import transforms

DATASET_DIR = f'../assets/coins_dataset'
MODEL_STATE_DIR = f'../model_state'

# Get the model for training
model, model_name = models.get_fasterrcnn_untrained()

# Transform images


class ResizeImage:
    def __call__(self, image: Image.Image) -> Image.Image:
        size = image.size
        # Resize the image to either 3000x4000 or 4000x3000 depending on the aspect ratio
        if size[0] > size[1]:
            image = image.resize((4000, 3000))
        else:
            image = image.resize((3000, 4000))

        return image


transform = transforms.Compose([
    ResizeImage(),
    data.default_transform,
])

# Get the data
data_loader_train, data_loader_test = data.get_data_loaders(DATASET_DIR)#, transform=transform)

# Train the model
training.train(model, model_name, data_loader_train, data_loader_test,
               state_dir=MODEL_STATE_DIR, print_frequency=25, num_epochs=100)
