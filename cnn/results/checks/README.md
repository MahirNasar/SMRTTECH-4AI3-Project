# M3 check: the GAP alarm catches the shot-chart shortcut

MA-4 "done when": training with the career shot chart visible in **both** views (a deliberate shortcut) must trigger the GAP alarm (validation normalized loss − train normalized loss > 0.30).
Date 2026-11-25, Apple M2, MPS, seed 0, early stopping on.

| Run | Command overrides | GAP alarm | Val top-1 at stop | Gap at stop |
|---|---|---|---|---|
| `gap_alarm_chart_same/` | `views.chart_mode=same` (chart noise 0.5 still on) | **epoch 14** | 0.40 | 0.74 |
| `gap_alarm_chart_same_nonoise/` | `views.chart_mode=same views.chart_noise=0` | **epoch 6** | 0.42 | 0.94 |

**Pass.** Both runs fire the alarm, and the gap keeps growing afterwards (train loss falls while validation loss rises back toward 1.0), which is the shortcut signature. Validation top-1 stalls at 0.40–0.42, matching the prototype's "identical shot chart" result (0.41, alarm at epoch 6).

The guide expects the alarm "within ~10 epochs". With `chart_noise` left at 0.5 it fires a little later (epoch 14), because the noise makes the identical chart a weaker cue. Without the noise, the shortcut is clean and it fires at epoch 6, exactly like the prototype. For comparison, the default config never fires it (`../first_run/baseline/`, max gap 0.04).
