import torch,pickle,sys,time,random,os
import torch.nn as nn
MODEL_PATH="genesis_brain.pth";VOCAB_PATH="genesis_vocab.pkl";HIDDEN_SIZE=256;NUM_LAYERS=2;TEMPERATURE=0.6;SPEED=0.05
if not os.path.exists(VOCAB_PATH):print(f"Bruh. Missing {VOCAB_PATH}. Did you run the training script?");sys.exit()
with open(VOCAB_PATH,"rb") as f:vd=pickle.load(f);char_to_ix,ix_to_char=vd["char_to_ix"],vd["ix_to_char"]
vocab_size=len(char_to_ix);print(f"--- VOCAB LOADED ({vocab_size} chars) ---")
def to_tensor(s):t=torch.zeros(len(s),dtype=torch.long);[t.__setitem__(i,char_to_ix.get(s[i],0)) for i in range(len(s))];return t
class TinyBrain(nn.Module):
    def __init__(self,vocab_size,hidden_size,num_layers):
        super().__init__();self.embedding=nn.Embedding(vocab_size,hidden_size);self.rnn=nn.LSTM(hidden_size,hidden_size,num_layers,batch_first=True);self.decoder=nn.Linear(hidden_size,vocab_size)
    def forward(self,x,h=None):
        e=self.embedding(x);o,h=self.rnn(e,h);return self.decoder(o),h
model=TinyBrain(vocab_size,HIDDEN_SIZE,NUM_LAYERS)
try:model.load_state_dict(torch.load(MODEL_PATH));model.eval();print("--- SIGMA WEIGHTS LOADED ---")
except Exception as e:print(f"Cringe. Could not load model: {e}");sys.exit()
@torch.no_grad()
def stream_generation(start_str,length=500,temp=0.8):
    h=None;inp=to_tensor(start_str).unsqueeze(0)
    for ch in start_str:sys.stdout.write(f"\033[92m{ch}\033[0m");sys.stdout.flush();time.sleep(random.uniform(SPEED*0.5,SPEED*1.5))
    for i in range(len(start_str)-1):_,h=model(inp[:,i:i+1],h)
    inp=inp[:,-1:].clone()
    for _ in range(length):
        o,h=model(inp,h);dist=o.view(-1).div(temp).exp();top_i=torch.multinomial(dist,1)[0];c=ix_to_char[top_i.item()];sys.stdout.write(f"\033[92m{c}\033[0m");sys.stdout.flush();d=SPEED
        if c in[".",",","!","?",":"]:d*=4
        elif c==" ":d*=1.5
        elif c=="\n":d*=3
        time.sleep(random.uniform(d*0.5,d*1.5));inp=to_tensor(c).unsqueeze(0)
    sys.stdout.write("\n")
while True:
    print("\n"+"="*40)
    try:prompt=input("\033[94mGive the Oracle a prompt (or Enter for default): \033[0m")
    except KeyboardInterrupt:print("\nExiting matrix...");break
    if not prompt:prompt="The King"
    print("-"*20);stream_generation(prompt,length=400,temp=TEMPERATURE)
