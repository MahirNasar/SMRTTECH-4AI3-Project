"""Time one training step on CPU and on MPS to decide which device to use."""
import time, torch
from nbacnn.fake import make_fake_split
from nbacnn.model import PlayerEncoder
from nbacnn.views import make_views
from nbacnn.losses import nt_xent
from nbacnn.device import sync

def bench(device, batch=128, steps=50):
    data = make_fake_split(batch).to(device)
    model = PlayerEncoder().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cfg = {"chart_mode": "drop_one"}
    for i in range(steps + 10):                     # first 10 steps are warm-up (not timed)
        if i == 10:
            sync(device); t0 = time.perf_counter()
        a, b = make_views(data, cfg)
        loss = nt_xent(model(*a)[1], model(*b)[1])
        opt.zero_grad(); loss.backward(); opt.step()
    sync(device)
    return (time.perf_counter() - t0) / steps * 1000

devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
for batch in (64, 128, 256):
    print(f"batch {batch:3d}: " + "  ".join(f"{d}={bench(torch.device(d), batch):6.1f} ms/step" for d in devices))
print(f"torch {torch.__version__}, threads {torch.get_num_threads()}")