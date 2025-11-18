import torch,requests,random,time,pickle
import torch.nn as nn,torch.optim as optim
HIDDEN_SIZE=256;NUM_LAYERS=2;LR=0.002;SEQ_LEN=100;EPOCHS=5000;PRINT_EVERY=250;TEMP=0.8;SAVE_PATH="genesis_brain.pth";VOCAB_PATH="genesis_vocab.pkl"
print("--- LOADING REALITY ---");url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
try:text=requests.get(url).text[:200000]
except:print("Bruh. No internet? Using fallback.");text="To be, or not to be, that is the question..."*1000
chars=sorted(list(set(text)));data_size,vocab_size=len(text),len(chars);print(f"Data: {data_size} chars | Vocab: {vocab_size} unique")
char_to_ix={ch:i for i,ch in enumerate(chars)};ix_to_char={i:ch for i,ch in enumerate(chars)}
with open(VOCAB_PATH,"wb") as f:pickle.dump({"char_to_ix":char_to_ix,"ix_to_char":ix_to_char},f);print(f"Vocabulary saved to {VOCAB_PATH}")
def to_tensor(s):t=torch.zeros(len(s),dtype=torch.long);[t.__setitem__(i,char_to_ix[s[i]]) for i in range(len(s))];return t
class TinyBrain(nn.Module):
    def __init__(self,vocab_size,hidden_size,num_layers):
        super().__init__();self.embedding=nn.Embedding(vocab_size,hidden_size);self.rnn=nn.LSTM(hidden_size,hidden_size,num_layers,batch_first=True);self.decoder=nn.Linear(hidden_size,vocab_size)
    def forward(self,x,h=None):
        e=self.embedding(x);o,h=self.rnn(e,h);return self.decoder(o),h
    def generate(self,start_str="A",predict_len=100,temperature=0.8):
        h=None;inp=to_tensor(start_str).unsqueeze(0);p=start_str
        for i in range(len(start_str)-1):_,h=self.forward(inp[:,i:i+1],h)
        inp=inp[:,-1:].clone()
        for _ in range(predict_len):
            o,h=self.forward(inp,h);dist=o.view(-1).div(temperature).exp();top_i=torch.multinomial(dist,1)[0];c=ix_to_char[top_i.item()];p+=c;inp=to_tensor(c).unsqueeze(0)
        return p
model=TinyBrain(vocab_size,HIDDEN_SIZE,NUM_LAYERS);opt=optim.Adam(model.parameters(),lr=LR);crit=nn.CrossEntropyLoss()
print("--- TRAINING STARTED ---");avg_loss=0.0
for epoch in range(1,EPOCHS+1):
    i=random.randint(0,data_size-SEQ_LEN-1);chunk=text[i:i+SEQ_LEN+1];inp=to_tensor(chunk[:-1]).unsqueeze(0);target=to_tensor(chunk[1:]).unsqueeze(0)
    opt.zero_grad();out,_=model(inp,None);loss=crit(out.view(-1,vocab_size),target.view(-1));loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();avg_loss+=loss.item()
    if epoch%PRINT_EVERY==0:print(f"Step {epoch}/{EPOCHS} | Loss: {avg_loss/PRINT_EVERY:.4f}");avg_loss=0.0
print("--- SAVING SIGMA STATE ---");torch.save(model.state_dict(),SAVE_PATH);print(f"Model weights saved to {SAVE_PATH}");print("Run 'genesis_speak.py' to hear it speak.")
