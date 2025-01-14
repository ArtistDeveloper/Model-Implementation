import numpy as np

import torch
import torch.nn as nn

class Test(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.label_emb = nn.Embedding(10, 10)
        print("self.label_emb type: ", type(self.label_emb))
        
    def forward(self, label):
        test = self.label_emb(label)
        
        return test
    

if __name__=="__main__":
    test_obj = Test()
    label = 16
    
    label_emb = test_obj(torch.IntTensor(label))
    
    print("label_emb: ", label_emb.shape)
    print(label_emb)
    