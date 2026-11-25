import torch
from nbacnn.fake import make_fake_split
from nbacnn.model import PlayerEncoder
from nbacnn.embed import embed_all, top_k

def test_embed_respects_chart_off_and_top_k_format():
    torch.manual_seed(0)
    model = PlayerEncoder()
    a, b = make_fake_split(20, seed=0), make_fake_split(20, seed=0)
    b.chart = b.chart * 5 + 3                                   # same players, different shot charts
    ea = embed_all(model, {"train": a}, {"chart_mode": "off"})
    eb = embed_all(model, {"train": b}, {"chart_mode": "off"})
    cols = [c for c in ea.columns if c.startswith("e") and c[1:].isdigit()]
    assert torch.allclose(torch.tensor(ea[cols].to_numpy()), torch.tensor(eb[cols].to_numpy()), atol=1e-5)
    nb = top_k(ea, ea.player_id.tolist(), k=5)
    assert list(nb.columns) == ["query_id", "rank", "neighbour_id", "similarity"]
    assert len(nb) == 100 and not (nb.query_id == nb.neighbour_id).any()
