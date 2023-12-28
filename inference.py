import argparse
from PIL import Image

import torch

import src.utils as utils
import src.dataset as dataset
import rare.seq2seq as crnn
import rare.transformer as stn

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, default='./data/test_img/20440250_792474788.jpg', help='the path of the input image to network')
parser.add_argument('--img_height', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--img_width', type=int, default=280, help='the width of the input image to network')
parser.add_argument('--hidden_size', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--transformer', type=str, default='./model/rare/transformer.pth', help='path to transformer')
parser.add_argument('--encoder', type=str, default='./model/rare/encoder.pth', help='path to encoder')
parser.add_argument('--decoder', type=str, default='./model/rare/decoder.pth', help='path to decoder')
parser.add_argument('--max_width', type=int, default=71, help='the width of the feature map out from cnn')
parser.add_argument('--use_gpu', action='store_true', help='whether use gpu')
cfg = parser.parse_args()


# load alphabet
with open('./data/char_std_5990.txt', "rt", encoding="utf-8") as f:
    data = f.readlines()
    alphabet = [x.rstrip() for x in data]
    alphabet = ''.join(alphabet)

# define convert between string and label index
converter = utils.ConvertBetweenStringAndLabel(alphabet)

# len(alphabet) + SOS_TOKEN + EOS_TOKEN
num_classes = len(alphabet) + 2

resizer = dataset.ResizeNormalize(img_width=cfg.img_width, img_height=cfg.img_height)


def seq2seq_decode(encoder_out, decoder, decoder_input, decoder_hidden, max_length):
    decoded_words = []
    prob = 1.0
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_out)
        probs = torch.exp(decoder_output)
        _, topi = decoder_output.data.topk(1)
        ni = topi.squeeze(1)
        decoder_input = ni
        prob *= probs[:, ni]
        if ni == utils.EOS_TOKEN:
            break
        else:
            decoded_words.append(converter.decode(ni))

    words = ''.join(decoded_words)
    prob = prob.item()

    return words, prob


def main():
    image = Image.open(cfg.img_path).convert('RGB')
    image = resizer(image)
    if torch.cuda.is_available() and cfg.use_gpu:
        image = image.cuda()
    image = image.view(1, *image.size())
    image = torch.autograd.Variable(image) 

    transformer = stn.STN(3, cfg.img_height, cfg.img_width, 0.9, 0.9, 4, 5)
    encoder = crnn.Encoder(3, cfg.hidden_size)
    # no dropout during inference
    decoder = crnn.Decoder(cfg.hidden_size, num_classes, dropout_p=0.0, max_length=cfg.max_width)

    if torch.cuda.is_available() and cfg.use_gpu:
        transformer = transformer.cuda()
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        map_location = 'cuda'
    else:
        map_location = 'cpu'

    transformer.load_state_dict(torch.load(cfg.transformer, map_location=map_location))
    print('loading pretrained transformer models from {}.'.format(cfg.transformer))
    encoder.load_state_dict(torch.load(cfg.encoder, map_location=map_location))
    print('loading pretrained encoder models from {}.'.format(cfg.encoder))
    decoder.load_state_dict(torch.load(cfg.decoder, map_location=map_location))
    print('loading pretrained decoder models from {}.'.format(cfg.decoder))

    transformer.eval()
    encoder.eval()
    decoder.eval()

    stn_output = transformer(image)
    # encoder_out = encoder(image)
    encoder_out = encoder(stn_output)

    max_length = 20
    decoder_input = torch.zeros(1).long()
    decoder_hidden = decoder.initHidden(1)
    if torch.cuda.is_available() and cfg.use_gpu:
        decoder_input = decoder_input.cuda()
        decoder_hidden = decoder_hidden.cuda()

    words, prob = seq2seq_decode(encoder_out, decoder, decoder_input, decoder_hidden, max_length)
    print('predict_string: {} => predict_probability: {}'.format(words, prob))

if __name__ == "__main__":
    main()