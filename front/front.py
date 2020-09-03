import torch
from common import *


if __name__ == "__main__":
    parser = Parser
    parser.add_argument(
        '--model1_path',
        type=str,
        default='./layer1',
        help="path to model 1"
    )

    model = layer1
    args = parser.parse_args()
    model.to(torch.device(args.device))
    model.eval()

    if args.model1_path is not None:
        model.load_state_dict(torch.load(args.model1_path))

    input_data = load_data(args.data)
    out = model(input_data[0])
    save_data(args.out_path, out)
