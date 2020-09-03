import torch
import torchvision
import torchvision.transforms as T
from common import *


if __name__ == "__main__":
    parser = Parser
    parser.add_argument(
        '--idx',
        type=int,
        default=0,
        help="index"
    )
    args = parser.parse_args()
    batch_size = args.batch
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = torchvision.datasets.CIFAR10('./cifar10_data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10('./cifar10_data', train=False, download=True, transform=transform)

    # Data loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(train_dataset.data.shape)
    print(test_dataset.data.shape)


    def convert_to_imshow_format(image):
        # first convert back to [0,1] range from [-1,1] range
        image = image / 2 + 0.5
        image = image.numpy()
        # convert from CHW to HWC
        # from 3x32x32 to 32x32x3
        return image.transpose(1, 2, 0)
    output = None

    if args.train:
        idx = 0
        for temp_output in train_loader:
            if idx < args.idx:
                pass
            else:
                output = temp_output
                break
            idx += 1
    else:
        idx = 0
        for temp_output in test_loader:
            if idx < args.idx:
                pass
            else:
                output = temp_output
            idx += 1

    save_data(args.out_path, output)
