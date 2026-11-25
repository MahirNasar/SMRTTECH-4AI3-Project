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

def test_halves_embedding_is_the_normalized_sum_of_both_halves():
    import torch.nn.functional as F
    from nbacnn.views import career_halves
    from nbacnn.embed import embed_players
    torch.manual_seed(0)
    model, data = PlayerEncoder().eval(), make_fake_split(16, seed=2)
    odd, even = career_halves(data)
    expected = F.normalize(model(*odd)[0] + model(*even)[0], dim=1)
    assert torch.allclose(embed_players(model, data), expected, atol=1e-6)
    assert torch.allclose(embed_players(model, data).norm(dim=1), torch.ones(16))

def test_unknown_embed_method_fails_loudly():
    import pytest
    from nbacnn.embed import embed_players
    with pytest.raises(ValueError):
        embed_players(PlayerEncoder().eval(), make_fake_split(4), method="whole")
