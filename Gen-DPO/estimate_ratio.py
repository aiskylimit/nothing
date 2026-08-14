import torch
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

MODEL_PATH = "/media/volume/ES_volumne/dat/Preference-Distillation-via-Value-based-Reinforcement-Learning/output/sft"
REF_PATH   = MODEL_PATH
DATA_PATH  = "/media/volume/ES_volumne/dat/Gen-DPO/datasets/ultra-feedback/train.jsonl"
N_SAMPLES  = 200
MAX_LEN    = 512
DEVICE     = "cuda:0"

class UltraFeedbackDataset(Dataset):
    def __init__(self, path, tokenizer, max_len, n_samples):
        self.samples, self.tokenizer, self.max_len = [], tokenizer, max_len
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= n_samples: break
                self.samples.append(json.loads(line))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        item     = self.samples[idx]
        prompt   = item["prompt"]
        chosen   = item["chosen"]
        rejected = item["rejected"]

        if isinstance(chosen, list):
            chosen   = chosen[-1]["content"]
            rejected = rejected[-1]["content"]

        plen = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])

        win_enc  = self.tokenizer(prompt + chosen,   return_tensors="pt",
                                  truncation=True, max_length=self.max_len)
        lose_enc = self.tokenizer(prompt + rejected, return_tensors="pt",
                                  truncation=True, max_length=self.max_len)
        return {
            "win_ids":  win_enc["input_ids"].squeeze(0),
            "lose_ids": lose_enc["input_ids"].squeeze(0),
            "win_mask": win_enc["attention_mask"].squeeze(0),
            "lose_mask":lose_enc["attention_mask"].squeeze(0),
            "plen":     plen,
        }

def collate_fn(batch):
    def pad(seqs, masks):
        L  = max(s.size(0) for s in seqs)
        ps = torch.zeros(len(seqs), L, dtype=torch.long)
        pm = torch.zeros(len(seqs), L, dtype=torch.long)
        for i, (s, m) in enumerate(zip(seqs, masks)):
            ps[i, :s.size(0)] = s
            pm[i, :m.size(0)] = m
        return ps, pm

    win_ids,  win_mask  = pad([b["win_ids"]  for b in batch], [b["win_mask"]  for b in batch])
    lose_ids, lose_mask = pad([b["lose_ids"] for b in batch], [b["lose_mask"] for b in batch])
    return {"win_ids": win_ids, "win_mask": win_mask,
            "lose_ids": lose_ids, "lose_mask": lose_mask,
            "plens": [b["plen"] for b in batch]}

@torch.no_grad()
def token_log_probs(model, ids, mask):
    logits = model(input_ids=ids, attention_mask=mask).logits
    lp     = torch.log_softmax(logits[:, :-1, :], dim=-1)
    tgt    = ids[:, 1:].unsqueeze(-1)
    return lp.gather(-1, tgt).squeeze(-1)

def main():
    print(f"Loading tokenizer from:\n  {MODEL_PATH}\n")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("Loading model (policy = ref = SFT checkpoint)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(DEVICE).eval()

    # policy = ref = SFT nên ratio ban đầu phải gần 1.0
    # Đây là prior distribution TRƯỚC khi DPO train
    ref = model   # same object, không tốn thêm VRAM

    loader = DataLoader(
        UltraFeedbackDataset(DATA_PATH, tok, MAX_LEN, N_SAMPLES),
        batch_size=4, collate_fn=collate_fn, shuffle=False
    )

    r_win, r_lose = [], []

    for batch in tqdm(loader, desc="Computing ratios"):
        wids  = batch["win_ids"].to(DEVICE)
        wmask = batch["win_mask"].to(DEVICE)
        lids  = batch["lose_ids"].to(DEVICE)
        lmask = batch["lose_mask"].to(DEVICE)

        lp_w  = token_log_probs(model, wids,  wmask)
        lp_l  = token_log_probs(model, lids,  lmask)

        # Vì policy = ref → log ratio = 0 → ratio = 1 everywhere
        # Nên mình tính token-level log prob difference giữa chosen vs rejected
        # để estimate range của r_t sau vài bước DPO đầu
        for i, plen in enumerate(batch["plens"]):
            w_lp = lp_w[i, plen:].cpu()
            l_lp = lp_l[i, plen:].cpu()

            # Proxy: ratio ~ exp(log_prob) chuẩn hóa để estimate scale
            w_probs = w_lp.exp().tolist()
            l_probs = l_lp.exp().tolist()

            r_win.extend( [p for p in w_probs if 0.0001 < p < 1.0])
            r_lose.extend([p for p in l_probs if 0.0001 < p < 1.0])

    r_win  = np.array(r_win)
    r_lose = np.array(r_lose)

    print("\n" + "="*55)
    print("Token-level π_sft(y^w_t) distribution (chosen tokens):")
    for p in [10, 25, 50, 75, 90, 95]:
        print(f"  p{p:02d} = {np.percentile(r_win,  p):.5f}")

    print("\nToken-level π_sft(y^l_t) distribution (rejected tokens):")
    for p in [5, 10, 25, 50, 75, 90]:
        print(f"  p{p:02d} = {np.percentile(r_lose, p):.5f}")

    # Estimate ε từ token prob distribution
    # Token prob thấp → model uncertain → ratio sẽ fluctuate mạnh hơn khi train
    # ε_h nên đủ lớn để không clip quá sớm trên uncertain tokens
    median_win  = np.percentile(r_win,  50)
    median_lose = np.percentile(r_lose, 50)

    print("\n" + "="*55)
    print(f"Median chosen   token prob : {median_win:.5f}")
    print(f"Median rejected token prob : {median_lose:.5f}")

    # Heuristic: token prob thấp → cần ε_h lớn hơn để học được
    if median_win < 0.05:
        eps_h = 0.25
    elif median_win < 0.15:
        eps_h = 0.20
    else:
        eps_h = 0.15

    # ε_l conservative hơn để tránh GSM8K forgetting
    if median_lose < 0.05:
        eps_l = 0.10
    elif median_lose < 0.15:
        eps_l = 0.08
    else:
        eps_l = 0.05

    print("\n" + "="*55)
    print(f"  => Recommended ε_h = {eps_h}")
    print(f"  => Recommended ε_l = {eps_l}")
    print("="*55)

if __name__ == "__main__":
    main()
