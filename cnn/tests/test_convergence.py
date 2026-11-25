from nbacnn.convergence import ConvergenceTracker

def test_best_checkpoint_follows_every_new_minimum_but_convergence_needs_min_delta():
    t = ConvergenceTracker(patience=3, min_delta=0.05)
    saves = [t.update(e, 0.5, v, 0.5) for e, v in enumerate([1.0, 0.9, 0.89, 0.88, 0.87, 0.86], start=1)]
    assert saves == [True] * 6                       # every small gain is a new best.pt
    assert t.best_ckpt_epoch == 6 and t.best_epoch == 2   # but only 1.0 -> 0.9 counts for convergence
    assert t.converged and t.converged_epoch == 5

def test_alarms_fire_once_per_kind():
    t = ConvergenceTracker(gap_alarm=0.3, too_good_top1=0.9)
    for e in range(1, 4):
        t.update(e, 0.1, 0.9, 0.95)
    assert len(t.alarms) == 2 and t.target_epoch == 1
