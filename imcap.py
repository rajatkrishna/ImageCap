from torchvision import transforms
from PIL import Image
import nltk
from vocabulary import Vocabulary
import torch
from model import EncoderCNN, DecoderRNN
import os
import glob

transform_img = transforms.Compose([transforms.Resize(256),
                                transforms.RandomCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406),
                                                     (0.229, 0.224, 0.225))])

embed_size = 256
vocab_threshold = 20
hidden_size = 512
encoder_file = "encoder.pkl"
decoder_file = "decoder.pkl"
#Image to be captioned
image_files = glob.glob('./images/*')

vocab = Vocabulary(vocab_threshold, vocab_from_file = True)
vocab_size = len(vocab)

encoder = EncoderCNN(embed_size)
encoder.eval()

decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
decoder.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.load_state_dict(torch.load(os.path.join('./models', encoder_file), map_location = device))
decoder.load_state_dict(torch.load(os.path.join('./models', decoder_file), map_location = device))


encoder.to(device)
decoder.to(device)

count = 0
for image_file in image_files:
    image = Image.open(image_file)
    
    image_copy = image.copy()
    image = transform_img(image)
    
    image = image.unsqueeze(0)

    image.to(device)

    features = encoder(image).unsqueeze(1)

    output = decoder.gen(features)

    sentence = ""
    for idx in range(1, len(output)):
        if output[(idx + 1) % len(output)] == vocab.word2idx['.']:
            sentence += vocab.idx2word[output[idx]]
        else:
            sentence += vocab.idx2word[output[idx]] + " "


#    image_copy.show(title = sentence)
    
    count += 1
    print("\n\nimage", count, "--> ", sentence)
    