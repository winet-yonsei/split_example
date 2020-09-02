import torch
from common import *


if __name__ == "__main__":
    parser = Parser
    parser.add_argument(
        '--out_data',
        type=str,
        default='./layer2_out.bin',
        help="Input data csv"
    )
    parser.add_argument(
        '--model2_path',
        default='./layer2',
        type=str,
        help="path to model 2"
    )

    args = parser.parse_args()
    layer2.to(torch.device(args.device))
    if args.model2_path is not None:
        layer2.load_state_dict(torch.load(args.model2_path))
    input_data = load_data(args.data)
    out = layer2(input_data)
    save_data(args.out_data, out)
