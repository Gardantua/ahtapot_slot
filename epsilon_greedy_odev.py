# Yunus Emre Arı
# Pekiştirmeli Öğrenme Ödevi — ε-Greedy ve Adaptive Epsilon Stratejileri
# Sadece numpy ve matplotlib kullandım, hazır RL kütüphanesi kullanmadım.

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import to_rgb
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from numpy.typing import NDArray
from typing import Optional
from abc import ABC, abstractmethod

CACHE_DIR = "cache"


def _cache_load(key: str) -> Optional[dict]:
    path = os.path.join(CACHE_DIR, f"{key}.npy")
    if os.path.exists(path):
        return np.load(path, allow_pickle=True).item()
    return None


def _cache_save(key: str, data: dict) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(os.path.join(CACHE_DIR, f"{key}.npy"), data)


# Deney parametreleri
# k=20 seçtim — 5 çok kolay, 50'de 10k adım yetmiyor; 20 tam orta nokta.
N_ARMS = 20
N_STEPS = 10_000
N_RUNS = 1_000
EPSILONS = (0.0, 0.01, 0.05, 0.1, 0.2, 0.4)
MASTER_SEED = 42


# --- Ortam ---

class KArmedBandit:
    """
    k-kollu bandit ortamı.
    q*(a) ~ N(0,1), ödül Rt ~ N(q*(a), 1).
    Varyans olmasaydı her kolu bir kez çekince gerçek değeri öğrenirdim,
    exploration-exploitation dengesi hiç önemli olmazdı.
    """

    def __init__(self, k: int = N_ARMS, seed: int = 0) -> None:
        self.k = k
        self.rng = np.random.default_rng(seed)
        # q*(a) ~ N(0, 1)
        self.q_star: NDArray[np.float64] = self.rng.standard_normal(self.k)
        self.optimal_action: int = int(np.argmax(self.q_star))

    def step(self, action: int) -> float:
        """Rt ~ N(q*(a), 1)"""
        # Rt ~ N(q*(a), 1)
        return float(self.rng.normal(loc=self.q_star[action], scale=1.0))


# --- Agent ---

class EpsilonGreedyAgent:
    """
    ε-Greedy: ε olasılıkla rastgele kol, (1-ε) olasılıkla argmax Qt(a).
    Tie-breaking: birden fazla kol aynı Q değerindeyse aralarından rastgele
    seçiyorum, yoksa hep aynı kola yönelip önyargı oluşuyor [S&B s.27].
    """

    def __init__(self, k: int, epsilon: float, seed: int = 0) -> None:
        self.k = k
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.Q: NDArray[np.float64] = np.zeros(k, dtype=np.float64)
        self.N: NDArray[np.int64] = np.zeros(k, dtype=np.int64)

    def select_action(self) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.k))
        else:
            # tie-breaking
            max_q = np.max(self.Q)
            max_arms = np.flatnonzero(self.Q == max_q)
            return int(self.rng.choice(max_arms))

    def update(self, action: int, reward: float) -> None:
        # Q(n+1) = Q(n) + (1/n) * [R(n) - Q(n)]
        # Geçmiş ödülleri saklamaya gerek kalmıyor, sample-average ile özdeş.
        self.N[action] += 1
        self.Q[action] += (1.0 / self.N[action]) * (reward - self.Q[action])

    def reset(self) -> None:
        self.Q[:] = 0.0
        self.N[:] = 0


# --- Adaptive Epsilon Stratejileri ---

class EpsilonSchedule(ABC):
    """Zamanla azalan epsilon için temel sınıf."""

    @abstractmethod
    def get_epsilon(self, step: int) -> float:
        ...


class LinearDecay(EpsilonSchedule):
    """
    εt = max(ε_min, ε0 - (ε0 - ε_min) / T_decay * t)
    ε0=1.0'dan başlayıp T_decay adımda ε_min=0.01'e düşüyor.
    T_decay = N_STEPS/2 seçtim: ilk yarıda explore, ikinci yarıda exploit.
    """

    def __init__(self, eps_start=1.0, eps_min=0.01, t_decay=N_STEPS // 2):
        self.eps_start = eps_start
        self.eps_min = eps_min
        self.t_decay = t_decay

    def get_epsilon(self, step: int) -> float:
        # εt = max(ε_min, ε0 - (ε0 - ε_min) / T_decay * t)
        eps = self.eps_start - (self.eps_start - self.eps_min) / self.t_decay * step
        return float(max(self.eps_min, eps))


class ExponentialDecay(EpsilonSchedule):
    """
    εt = max(ε_min, ε0 · d^t)
    d=0.9991 seçtim: 0.9991^5000 ≈ 0.011 ≈ ε_min, yani t=5000'de platoya oturuyor.
    Başta hızlı düşüyor, erken adımlarda linear'dan daha iyi öğreniyor.
    """

    def __init__(self, eps_start=1.0, eps_min=0.01, d=0.9991):
        self.eps_start = eps_start
        self.eps_min = eps_min
        self.d = d

    def get_epsilon(self, step: int) -> float:
        # εt = max(ε_min, ε0 · d^t)
        eps = self.eps_start * (self.d ** step)
        return float(max(self.eps_min, eps))


class InverseDecay(EpsilonSchedule):
    """
    εt = c / (c + t)
    t=0'da ε=1.0, sonra azalıyor ama hiç sıfıra inmiyor.
    c=500: t=500'de ε≈0.5, t=4500'de ε≈0.1.
    Teorik olarak logaritmik regret garantisi var.
    """

    def __init__(self, c=500.0):
        self.c = c

    def get_epsilon(self, step: int) -> float:
        # εt = c / (c + t)
        return float(self.c / (self.c + step))


# --- Deney ---

class Experiment:
    """1000 run × 10000 adım deneyi çalıştırır, ortalamaları döndürür."""

    def __init__(self, n_arms=N_ARMS, n_steps=N_STEPS, n_runs=N_RUNS):
        self.n_arms = n_arms
        self.n_steps = n_steps
        self.n_runs = n_runs

    def run_fixed_epsilon(self, epsilon: float, seed: int = MASTER_SEED) -> dict:
        rewards = np.zeros(self.n_steps)
        optimal_pct = np.zeros(self.n_steps)
        cum_regret = np.zeros(self.n_steps)

        rng = np.random.default_rng(seed)

        for run in range(self.n_runs):
            run_seed = int(rng.integers(0, 2**31))
            bandit = KArmedBandit(k=self.n_arms, seed=run_seed)
            agent = EpsilonGreedyAgent(k=self.n_arms, epsilon=epsilon, seed=run_seed + 1)
            regret = 0.0

            for t in range(self.n_steps):
                action = agent.select_action()
                reward = bandit.step(action)
                agent.update(action, reward)

                # regret += q*(a*) - q*(At)
                regret += bandit.q_star[bandit.optimal_action] - bandit.q_star[action]

                rewards[t] += reward
                optimal_pct[t] += 1.0 if action == bandit.optimal_action else 0.0
                cum_regret[t] += regret

        rewards /= self.n_runs
        optimal_pct /= self.n_runs / 100.0
        cum_regret /= self.n_runs

        return {"rewards": rewards, "optimal_pct": optimal_pct, "cum_regret": cum_regret}

    def run_adaptive_epsilon(self, schedule: EpsilonSchedule, seed: int = MASTER_SEED) -> dict:
        rewards = np.zeros(self.n_steps)
        optimal_pct = np.zeros(self.n_steps)
        cum_regret = np.zeros(self.n_steps)

        rng = np.random.default_rng(seed)

        for run in range(self.n_runs):
            run_seed = int(rng.integers(0, 2**31))
            bandit = KArmedBandit(k=self.n_arms, seed=run_seed)
            agent = EpsilonGreedyAgent(k=self.n_arms, epsilon=0.0, seed=run_seed + 1)
            regret = 0.0

            for t in range(self.n_steps):
                agent.epsilon = schedule.get_epsilon(t)
                action = agent.select_action()
                reward = bandit.step(action)
                agent.update(action, reward)

                regret += bandit.q_star[bandit.optimal_action] - bandit.q_star[action]

                rewards[t] += reward
                optimal_pct[t] += 1.0 if action == bandit.optimal_action else 0.0
                cum_regret[t] += regret

        rewards /= self.n_runs
        optimal_pct /= self.n_runs / 100.0
        cum_regret /= self.n_runs

        return {"rewards": rewards, "optimal_pct": optimal_pct, "cum_regret": cum_regret}


# --- Grafik ---

# Paul Tol Bright paleti — renk körlüğüne uyumlu
EPSILON_COLORS = {
    0.00: "#444444",
    0.01: "#4477AA",
    0.05: "#228833",
    0.10: "#CC3311",
    0.20: "#EE7733",
    0.40: "#AA3377",
}

ADAPTIVE_COLORS = {
    "Linear Decay":      "#0077BB",
    "Exponential Decay": "#009988",
    "Inverse Decay":     "#EE3377",
    "Best Fixed":        "#CC3311",
}

BG        = "#FFFFFF"
PANEL     = "#F8F8F8"
GRID_COLOR = "#E0E0E0"
TEXT      = "#1A1A1A"
SUBTEXT   = "#666666"
SPINE_COL = "#CCCCCC"


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "figure.dpi":        150,
        "figure.figsize":    (10, 5.5),
        "axes.facecolor":    PANEL,
        "axes.edgecolor":    SPINE_COL,
        "axes.labelcolor":   TEXT,
        "axes.titlecolor":   TEXT,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "axes.titlepad":     12,
        "axes.labelsize":    10,
        "axes.labelweight":  "bold",
        "axes.labelpad":     7,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "axes.axisbelow":    True,
        "grid.color":        GRID_COLOR,
        "grid.linestyle":    "-",
        "grid.linewidth":    0.8,
        "lines.linewidth":   2.0,
        "xtick.color":       SUBTEXT,
        "ytick.color":       SUBTEXT,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "legend.facecolor":  BG,
        "legend.edgecolor":  SPINE_COL,
        "legend.labelcolor": TEXT,
        "legend.fontsize":   9,
        "font.family":       "sans-serif",
        "text.color":        TEXT,
        "savefig.facecolor": BG,
        "savefig.bbox":      "tight",
        "savefig.dpi":       220,
    })


def _gradient_fill(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str, alpha: float = 0.12) -> None:
    """Eğri altına şeffaf gradient dolgu."""
    rgb = to_rgb(color)
    gradient = np.zeros((256, 1, 4))
    gradient[:, 0, :3] = rgb
    gradient[:, 0, 3] = np.linspace(alpha, 0, 256)
    im = ax.imshow(
        gradient, aspect="auto",
        extent=[x[0], x[-1], float(np.min(y)), float(np.max(y))],
        origin="upper", zorder=1,
    )
    verts = [(x[0], float(np.min(y)))] + list(zip(x, y)) + \
            [(x[-1], float(np.min(y))), (x[0], float(np.min(y)))]
    codes = [Path.MOVETO] + [Path.LINETO] * len(x) + [Path.LINETO, Path.CLOSEPOLY]
    im.set_clip_path(PathPatch(Path(verts, codes), transform=ax.transData))


def _finalize(ax: plt.Axes, fig: plt.Figure, footnote: str) -> None:
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color(SPINE_COL)
    ax.tick_params(length=3, width=0.8)
    fig.text(0.99, 0.012, footnote, ha="right", va="bottom",
             fontsize=7.5, color=SUBTEXT, style="italic")
    plt.tight_layout()


def plot_average_reward(steps: np.ndarray, results: dict, save_path: str = "grafik_1_ortalama_odul.png") -> None:
    apply_style()
    fig, ax = plt.subplots()

    best_eps = max(results, key=lambda e: np.mean(results[e]["rewards"][-500:]))

    for eps, data in results.items():
        r = data["rewards"]
        color = EPSILON_COLORS[eps]
        lw = 2.4 if eps == best_eps else 1.6
        label = f"e={eps:.2f}" + ("  *" if eps == best_eps else "")
        ax.plot(steps, r, color=color, linewidth=lw, label=label, zorder=3)
        _gradient_fill(ax, steps, r, color, alpha=0.18 if eps == best_eps else 0.07)

    idx = len(steps) - 500
    ax.annotate(
        f"En iyi: e={best_eps:.2f}",
        xy=(steps[idx], results[best_eps]["rewards"][idx]),
        xytext=(steps[idx] - len(steps) * 0.2, results[best_eps]["rewards"][idx] * 0.88),
        fontsize=8.5, color=EPSILON_COLORS[best_eps], fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=EPSILON_COLORS[best_eps],
                        lw=1.2, connectionstyle="arc3,rad=0.2"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor=BG,
                  edgecolor=EPSILON_COLORS[best_eps], lw=0.8),
    )

    all_rewards = np.array([d["rewards"] for d in results.values()])
    ax.set_ylim(all_rewards.min() * 1.1, all_rewards.max() * 1.15)
    ax.set_xlim(0, steps[-1] * 1.02)
    ax.set_title("Ortalama Odul — Zaman Adimina Gore")
    ax.set_xlabel("Zaman Adimi  (t)")
    ax.set_ylabel("Ortalama Odul  E[Rt]")
    ax.legend(title="Epsilon", loc="lower right", ncol=2)
    _finalize(ax, fig, "1000 run ortalaması · k=20 · T=10 000")
    fig.savefig(save_path)
    print(f"[OK] {save_path}")


def plot_optimal_action(steps: np.ndarray, results: dict, k: int = N_ARMS, save_path: str = "grafik_2_optimal_eylem.png") -> None:
    apply_style()
    fig, ax = plt.subplots()

    for eps in [0.1, 0.2]:
        if eps not in results:
            continue
        # Teorik üst sınır: (1-ε)*100 + ε*(100/k)
        t_max = (1 - eps) * 100 + eps * (100 / k)
        ax.axhline(t_max, color=EPSILON_COLORS[eps], linestyle="--", linewidth=0.9, alpha=0.5)
        ax.text(steps[-1] * 1.001, t_max, f" {t_max:.1f}%",
                va="center", fontsize=7.5, color=EPSILON_COLORS[eps], alpha=0.75)

    for eps, data in results.items():
        p = data["optimal_pct"]
        color = EPSILON_COLORS[eps]
        ax.plot(steps, p, color=color, linewidth=1.8, label=f"e={eps:.2f}", zorder=2)
        _gradient_fill(ax, steps, p, color, alpha=0.08)

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylim(0, 108)
    ax.set_title("Optimal Eylem Orani — Zaman Adimina Gore")
    ax.set_xlabel("Zaman Adimi  (t)")
    ax.set_ylabel("Optimal Eylem (%)")
    ax.legend(title="Epsilon", loc="lower right", ncol=2)
    _finalize(ax, fig, "Kesikli cizgi: teorik ust sinir  (1-e)*100 + e*(100/k)")
    fig.savefig(save_path)
    print(f"[OK] {save_path}")


def plot_cumulative_regret(steps: np.ndarray, results: dict, save_path: str = "grafik_3_kumulatif_regret.png") -> None:
    apply_style()
    fig, ax = plt.subplots()

    best_eps = min(results, key=lambda e: results[e]["cum_regret"][-1])
    worst_eps = max(results, key=lambda e: results[e]["cum_regret"][-1])

    for eps, data in results.items():
        r = data["cum_regret"]
        color = EPSILON_COLORS[eps]
        lw = 2.2 if eps in [best_eps, worst_eps] else 1.6
        ax.plot(steps, r, color=color, linewidth=lw, label=f"e={eps:.2f}", zorder=2)
        _gradient_fill(ax, steps, r, color, alpha=0.07)

    ax.set_title("Kumulatif Pismanlık — Zaman Adimina Gore")
    ax.set_xlabel("Zaman Adimi  (t)")
    ax.set_ylabel("Kumulatif Regret")
    ax.legend(title="Epsilon", loc="upper left", ncol=2)
    _finalize(ax, fig, "Regret(t) = toplam[ q*(a*) - q*(At) ]")
    fig.savefig(save_path)
    print(f"[OK] {save_path}")


def plot_adaptive_comparison(steps: np.ndarray, fixed_best_rewards: np.ndarray, fixed_best_optimal: np.ndarray,
                              adaptive_results: dict, save_path: str = "grafik_4_adaptive_karsilastirma.png") -> None:
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    ax1.plot(steps, fixed_best_rewards, color=ADAPTIVE_COLORS["Best Fixed"],
             linewidth=1.6, linestyle="--", label="En iyi sabit e (referans)", alpha=0.8)
    _gradient_fill(ax1, steps, fixed_best_rewards, ADAPTIVE_COLORS["Best Fixed"], alpha=0.04)

    for name, data in adaptive_results.items():
        color = ADAPTIVE_COLORS[name]
        ax1.plot(steps, data["rewards"], color=color, linewidth=2.2, label=name)
        _gradient_fill(ax1, steps, data["rewards"], color, alpha=0.12)

    all_r = np.array([d["rewards"] for d in adaptive_results.values()] + [fixed_best_rewards])
    ax1.set_ylim(all_r.min() * 1.1, all_r.max() * 1.15)
    ax1.set_xlim(0, steps[-1] * 1.02)
    ax1.set_title("Ortalama Odul")
    ax1.set_xlabel("Zaman Adimi  (t)")
    ax1.set_ylabel("Ortalama Odul  E[Rt]")
    ax1.legend(title="Strateji", loc="lower right", fontsize=8)

    ax2.plot(steps, fixed_best_optimal, color=ADAPTIVE_COLORS["Best Fixed"],
             linewidth=1.6, linestyle="--", label="En iyi sabit e (referans)", alpha=0.8)
    _gradient_fill(ax2, steps, fixed_best_optimal, ADAPTIVE_COLORS["Best Fixed"], alpha=0.04)

    for name, data in adaptive_results.items():
        color = ADAPTIVE_COLORS[name]
        ax2.plot(steps, data["optimal_pct"], color=color, linewidth=2.2, label=name)
        _gradient_fill(ax2, steps, data["optimal_pct"], color, alpha=0.12)

    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax2.set_ylim(0, 108)
    ax2.set_xlim(0, steps[-1] * 1.02)
    ax2.set_title("Optimal Eylem Orani")
    ax2.set_xlabel("Zaman Adimi  (t)")
    ax2.set_ylabel("Optimal Eylem (%)")
    ax2.legend(title="Strateji", loc="lower right", fontsize=8)

    fig.suptitle("Adaptive Stratejiler — Karsilastirma (Gorev 7)",
                 fontsize=13, fontweight="bold", color=TEXT, y=1.01)
    for spine in ["left", "bottom"]:
        ax1.spines[spine].set_linewidth(0.8)
        ax2.spines[spine].set_linewidth(0.8)
    fig.text(0.99, 0.012, "Kesikli cizgi: referans (en iyi sabit e) · 1000 run",
             ha="right", va="bottom", fontsize=7.5, color=SUBTEXT, style="italic")
    plt.tight_layout()
    fig.savefig(save_path)
    print(f"[OK] {save_path}")


def plot_k_comparison(steps: np.ndarray, k_results: dict, save_path: str = "grafik_5_k_karsilastirma.png") -> None:
    apply_style()
    K_COLORS = {5: "#0077BB", 20: "#CC3311", 50: "#009988"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for k, data in k_results.items():
        color = K_COLORS[k]
        ax1.plot(steps, data["rewards"], color=color, linewidth=2.0, label=f"k={k}")
        _gradient_fill(ax1, steps, data["rewards"], color, alpha=0.10)
        ax2.plot(steps, data["optimal_pct"], color=color, linewidth=2.0, label=f"k={k}")
        _gradient_fill(ax2, steps, data["optimal_pct"], color, alpha=0.10)

    ax1.set_title("Ortalama Odul")
    ax1.set_xlabel("Zaman Adimi  (t)")
    ax1.set_ylabel("Ortalama Odul  E[Rt]")
    ax1.legend(title="Kol Sayisi", loc="lower right")

    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax2.set_ylim(0, 108)
    ax2.set_title("Optimal Eylem Orani")
    ax2.set_xlabel("Zaman Adimi  (t)")
    ax2.set_ylabel("Optimal Eylem (%)")
    ax2.legend(title="Kol Sayisi", loc="lower right")

    fig.suptitle("Kol Sayisinin Etkisi — k=5 vs k=20 vs k=50  (e=0.1)",
                 fontsize=13, fontweight="bold", color=TEXT, y=1.01)
    for spine in ["left", "bottom"]:
        ax1.spines[spine].set_linewidth(0.8)
        ax2.spines[spine].set_linewidth(0.8)
    fig.text(0.99, 0.012, "Sabit e=0.1 · 1000 run ortalaması",
             ha="right", va="bottom", fontsize=7.5, color=SUBTEXT, style="italic")
    plt.tight_layout()
    fig.savefig(save_path)
    print(f"[OK] {save_path}")


def plot_epsilon_schedules(steps: np.ndarray, save_path: str = "grafik_72_epsilon_degisim.png") -> None:
    """Her stratejinin εt değerini doğrudan hesaplıyor, simülasyon yok."""
    apply_style()
    fig, ax = plt.subplots()

    schedules = {
        "Linear Decay":      LinearDecay(),
        "Exponential Decay": ExponentialDecay(),
        "Inverse Decay":     InverseDecay(),
    }

    for name, sched in schedules.items():
        eps_vals = np.array([sched.get_epsilon(t) for t in steps])
        ax.plot(steps, eps_vals, color=ADAPTIVE_COLORS[name], linewidth=2.2, label=name)

    ax.set_title("Adaptive Epsilon Stratejileri — et Zamanla Degisimi (Grafik 7.2)")
    ax.set_xlabel("Zaman Adimi  (t)")
    ax.set_ylabel("Epsilon  (et)")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(title="Strateji", loc="upper right")
    _finalize(ax, fig, "Gercek et degerleri — simulasyon gerekmez")
    fig.savefig(save_path)
    print(f"[OK] {save_path}")


def plot_sensitivity_analysis(steps: np.ndarray, sensitivity_results: dict, param_name: str,
                               save_path: str = "grafik_73_duyarlilik.png") -> None:
    """T_decay değiştirince performans nasıl değişiyor."""
    apply_style()
    SENS_COLORS = ["#0077BB", "#CC3311", "#009988", "#EE7733", "#AA3377"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for i, (label, data) in enumerate(sensitivity_results.items()):
        color = SENS_COLORS[i % len(SENS_COLORS)]
        ax1.plot(steps, data["rewards"], color=color, linewidth=2.0, label=label)
        ax2.plot(steps, data["optimal_pct"], color=color, linewidth=2.0, label=label)

    ax1.set_title("Ortalama Odul")
    ax1.set_xlabel("Zaman Adimi  (t)")
    ax1.set_ylabel("Ortalama Odul  E[Rt]")
    ax1.legend(title=param_name, loc="lower right", fontsize=8)

    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax2.set_ylim(0, 108)
    ax2.set_title("Optimal Eylem Orani")
    ax2.set_xlabel("Zaman Adimi  (t)")
    ax2.set_ylabel("Optimal Eylem (%)")
    ax2.legend(title=param_name, loc="lower right", fontsize=8)

    fig.suptitle(
        f"Parametre Duyarlilik Analizi — Linear Decay, {param_name} degisimi (Grafik 7.3)",
        fontsize=12, fontweight="bold", color=TEXT, y=1.01,
    )
    for spine in ["left", "bottom"]:
        ax1.spines[spine].set_linewidth(0.8)
        ax2.spines[spine].set_linewidth(0.8)
    fig.text(0.99, 0.012, "Linear Decay · e0=1.0 sabit · 1000 run",
             ha="right", va="bottom", fontsize=7.5, color=SUBTEXT, style="italic")
    plt.tight_layout()
    fig.savefig(save_path)
    print(f"[OK] {save_path}")


# --- Main ---

def main() -> None:
    exp = Experiment()
    steps = np.arange(N_STEPS)

    # Sabit epsilon deneyleri
    print("Sabit epsilon deneyleri calistiriliyor...")
    fixed_results = {}
    for eps in EPSILONS:
        key = f"fixed_eps_{eps:.2f}"
        cached = _cache_load(key)
        if cached is not None:
            fixed_results[eps] = cached
            print(f"  e={eps:.2f} cache'den yuklendi")
        else:
            print(f"  e={eps:.2f} hesaplaniyor...", end=" ", flush=True)
            fixed_results[eps] = exp.run_fixed_epsilon(epsilon=eps)
            _cache_save(key, fixed_results[eps])
            print("tamam")

    plot_average_reward(steps, fixed_results)
    plot_optimal_action(steps, fixed_results)
    plot_cumulative_regret(steps, fixed_results)

    # Adaptive epsilon deneyleri
    print("\nAdaptive epsilon deneyleri calistiriliyor...")
    schedules = {
        "Linear Decay":      LinearDecay(),
        "Exponential Decay": ExponentialDecay(),
        "Inverse Decay":     InverseDecay(),
    }

    adaptive_results = {}
    for name, schedule in schedules.items():
        key = f"adaptive_{name.replace(' ', '_').lower()}"
        cached = _cache_load(key)
        if cached is not None:
            adaptive_results[name] = cached
            print(f"  {name} cache'den yuklendi")
        else:
            print(f"  {name} hesaplaniyor...", end=" ", flush=True)
            result = exp.run_adaptive_epsilon(schedule=schedule)
            adaptive_results[name] = {
                "rewards":     result["rewards"],
                "optimal_pct": result["optimal_pct"],
            }
            _cache_save(key, adaptive_results[name])
            print("tamam")

    best_eps = max(fixed_results, key=lambda e: np.mean(fixed_results[e]["rewards"][-500:]))
    plot_adaptive_comparison(steps, fixed_results[best_eps]["rewards"],
                             fixed_results[best_eps]["optimal_pct"], adaptive_results)

    # k karşılaştırması
    print("\nk karsilastirma deneyleri calistiriliyor (e=0.1)...")
    k_results = {}
    for k_val in [5, 20, 50]:
        key = f"k_comparison_k{k_val}"
        cached = _cache_load(key)
        if cached is not None:
            k_results[k_val] = cached
            print(f"  k={k_val} cache'den yuklendi")
        else:
            print(f"  k={k_val} hesaplaniyor...", end=" ", flush=True)
            k_results[k_val] = Experiment(n_arms=k_val).run_fixed_epsilon(epsilon=0.1)
            _cache_save(key, k_results[k_val])
            print("tamam")

    plot_k_comparison(steps, k_results)
    plot_epsilon_schedules(steps)

    # Parametre duyarlılık analizi
    print("\nDuyarlilik analizi (T_decay degisimi) calistiriliyor...")
    tdecay_results = {}
    for t_val in [500, 2000, 5000]:
        label = f"T_decay={t_val}"
        key = f"sensitivity_tdecay{t_val}"
        cached = _cache_load(key)
        if cached is not None:
            tdecay_results[label] = cached
            print(f"  {label} cache'den yuklendi")
        else:
            print(f"  {label} hesaplaniyor...", end=" ", flush=True)
            sched = LinearDecay(eps_start=1.0, eps_min=0.01, t_decay=t_val)
            result = exp.run_adaptive_epsilon(schedule=sched)
            tdecay_results[label] = {"rewards": result["rewards"], "optimal_pct": result["optimal_pct"]}
            _cache_save(key, tdecay_results[label])
            print("tamam")

    plot_sensitivity_analysis(steps, tdecay_results, param_name="T_decay")

    print("\nTamamlandi.")


if __name__ == "__main__":
    main()
