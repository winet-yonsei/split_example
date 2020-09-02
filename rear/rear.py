import torch
from common import *


if __name__ == "__main__":
    parser = Parser
    parser.add_argument(
        '--out_data',
        type=str,
        default='./fc_out.bin',
        help="Input data csv"
    )
    parser.add_argument(
        '--fc_path',
        type=str,
        default='./fc',
        help="path to fc"
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=10,
        help='number of class'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='path to image'
    )

    args = parser.parse_args()
    fc = make_fc_layer(args.num_classes)
    fc.to(torch.device(args.device))
    fc.eval()

    if args.fc_path is not None:
        fc.load_state_dict(torch.load(args.fc_path))
    input_data = load_data(args.data)
    input_data = input_data.reshape(input_data.size(0), -1)
    out = fc(input_data)

    save_data(args.out_data, out)
    output = torch.argmax(out, 1)

    input_image = load_data(args.image)
    target = input_image[1]
    print("Test accuracy: %s" % str(((output == target).sum().item()) / output.shape[0]))
