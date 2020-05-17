import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        #load pretrained resnet152 architecture
        resnet = models.resnet152(pretrained=True)

        for param in resnet.parameters():
            param.requires_grad_(False)
        
        #remove last layer and add embedding layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(input_size = embed_size, hidden_size = hidden_size, num_layers = num_layers, bias = True, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        captions = captions[:, :-1]
        
        embedding = self.word_embedding(captions)
        
        embedding = torch.cat((features.unsqueeze(1), embedding), dim=1)
        out, _= self.lstm(embedding)
        
        outputs = self.linear(out)
        return outputs

    def gen(self, inputs, states=None, max_len=20):
        length = 0
        output = []
        
        while length <= max_len:
            out, states = self.lstm(inputs, states)

            out = out.squeeze(dim = 1)
            out = self.linear(out)
            _, pred = torch.max(out, 1)
            if pred == 1:
                break
            output.append(int(pred))
            
            
            inputs = self.word_embedding(pred)
            inputs = inputs.unsqueeze(dim = 1)

            length += 1

        return output