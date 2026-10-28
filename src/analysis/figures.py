"""Figure 1: Qwen buyer support across human vote splits (retail marketplace)."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

STYLE = {'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 10,
         'axes.spines.top': False, 'axes.spines.right': False,
         'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': .7,
         'savefig.facecolor': 'white'}


def vote_split_figure(d, path_stem):
    """Buyer-support rate with 95% Wilson intervals by the number of buyer-supporting
    jurors (out of 17), the real outcome (17-juror majority) as a dashed step, and group
    sizes in the lower panel. Returns the per-vote summary."""
    plt.rcParams.update(STYLE)
    s = (d.assign(qwen=d['Qwen|C0'].astype(int))
         .groupby('buyer_votes').agg(n=('qwen', 'size'), rate=('qwen', 'mean')).reset_index())
    z = stats.norm.ppf(.975)
    n, p = s.n.to_numpy(), s.rate.to_numpy()
    center = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(6.5, 3.9), sharex=True,
                                 gridspec_kw={'height_ratios': [3.5, 1], 'hspace': .08})
    ax.errorbar(s.buyer_votes, 100 * p, yerr=np.vstack([100 * (p - center + half), 100 * (center + half - p)]),
                fmt='o-', color='black', lw=1, ms=3.5, capsize=2, label='Qwen buyer support (95% CI)')
    ax.step([-.5, 8.5, 17.5], [0, 100, 100], where='post', color='.55', ls='--',
            label='Real outcome (platform majority)')
    ax.set_ylim(-4, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel('Buyer support (%)')
    ax.legend(loc='upper left', frameon=False, fontsize=8.5)
    bx.bar(s.buyer_votes, n, color='.65', width=.7)
    bx.set_ylabel('Cases')
    bx.set_xlabel('Number of buyer-supporting human jurors (out of 17)')
    bx.set_xticks(range(18))
    bx.set_xlim(-.5, 17.5)
    fig.align_ylabels()
    fig.subplots_adjust(left=.12, right=.98, bottom=.14, top=.97)
    fig.savefig(f'{path_stem}.pdf')
    fig.savefig(f'{path_stem}.png', dpi=220)
    plt.close(fig)
    s['wilson_low'], s['wilson_high'] = center - half, center + half
    return s
