"""
Graphene Oxide Neural Network Simulation
=========================================
Pipeline:
  1. Load I-V curve files (one per reduction state)
  2. Extract conductance G = I/V for each state (Ohm's law)
  3. Normalize conductances -> discrete weight levels
  4. Train a 2-layer neural net on MNIST, weights constrained to GO states
  5. Plot: I-V curves, weight levels, training curve, accuracy vs n_states
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

IV_FILES = [
    # 0 min
    "0b1.csv",
    "0m1.csv",
    "0t1.csv",
    # 10 min
    "10t1.csv",
    # 15 min
    "15b1.csv",
    "15m1.csv",
    # 45 min
    "45b1.csv",
    "45m1.csv",
    "45t1.csv",
    # 60 min
    "60b1.csv",
    "60m1.csv",
    "60t1.csv",
    # 90 min
    "90b1.csv",
    "90b2.csv",
    "90m1.csv",
    "90m2.csv",
    "90t1.csv",
    "90t2.csv",
    # 120 min
    "120b1.csv",
    "120b2.csv",
    "120m1.csv",
    "120m2.csv",
    "120t1.csv",
    "120t2.csv",
    # 150 min
    "150b1.csv",
    "150m1.csv",
    "150t1.csv",
    # 180 min
    "180b1.csv",
    "180b2.csv",
    "180m1.csv",
    "180m2.csv",
    "180t1.csv",
    "180t2.csv",
    # 210 min
    "210b1.csv",
    "210b2.csv",
    "210m1.csv",
    "210m2.csv",
    "210t1.csv",
    "210t2.csv",
    # 240 min
    "240b1.csv",
    "240b2.csv",
    "240m1.csv",
    "240m2.csv",
    "240t1.csv",
    "240t2.csv",
    # 270 min
    "270b1.csv",
    "270b2.csv",
    "270m1.csv",
    "270m2.csv",
    "270t1.csv",
    "270t2.csv",
    # 300 min
    "300b1.csv",
    "300b2.csv",
    "300m1.csv",
    "300m2.csv",
    "300t1.csv",
    "300t2.csv",
    # 330 min
    "330b1.csv",
    "330b2.csv",
    "330m1.csv",
    "330m2.csv",
    "330t1.csv",
    "330t2.csv",
    # 360 min
    "360b1.csv",
    "360b2.csv",
    "360m1.csv",
    "360t1.csv",
    "360t2.csv",
]

CURRENT_UNIT = "A"    # your currents are in amps (e.g. 8.5e-08)
VOLTAGE_COL  = 0      # after stripping the index column
CURRENT_COL  = 1      # after stripping the index column

# How to extract a single G value from a full I-V curve:
#   "slope"  = linear regression slope (best for ohmic devices)
#   "point"  = G at a specific voltage (set EVAL_VOLTAGE below)
CONDUCTANCE_METHOD = "slope"
EVAL_VOLTAGE = 0.5

# Set to True to also run the accuracy vs n_states sweep (takes ~5 min extra)
RUN_STATE_SWEEP = True

# =============================================================================
# STEP 1 & 2: Load I-V files and extract conductance
# =============================================================================

def load_iv_files(file_list):
    """Load CSV files, return list of (voltage_array, current_array_in_A).
    Expects format: index, voltage, current  (with 1 header row)
    """
    curves = []
    for f in file_list:
        try:
            # usecols=(1,2) skips the leading row-index column
            data = np.loadtxt(f, delimiter=",", skiprows=1, usecols=(1, 2))
            v = data[:, 0]  # voltage
            i = data[:, 1]  # current (already in amps)
            curves.append((v, i))
        except Exception as e:
            print(f"  WARNING: could not load {f}: {e}")
    return curves

def extract_conductance(v, i, method="slope", eval_v=0.5):
    """Extract a single G (Siemens) from one I-V curve."""
    if method == "slope":
        # Linear regression through origin: I = G*V
        g = np.dot(v, i) / np.dot(v, v)
        return abs(g)
    elif method == "point":
        i_at_v = np.interp(eval_v, v, i)
        return abs(i_at_v / eval_v) if eval_v != 0 else 0.0

# =============================================================================
# STEP 3: Normalize conductances to weight levels in [0, 1]
# =============================================================================

def conductance_to_weights(g_values):
    """Rank-based normalization centered around zero, giving range [-1, 1]."""
    g = np.array(g_values)
    ranks = np.argsort(np.argsort(g))
    normalized = ranks / (len(ranks) - 1)  # 0 to 1
    return (normalized * 2) - 1             # shift to -1 to +1

def quantize_to_go_states(w, go_weight_levels):
    """Snap each weight to the nearest available GO conductance level."""
    levels = np.array(go_weight_levels)
    dists  = np.abs(w[..., np.newaxis] - levels)
    idx    = np.argmin(dists, axis=-1)
    return levels[idx]

# =============================================================================
# STEP 4: MNIST neural network
# =============================================================================

def load_mnist():
    """Load MNIST via keras (standalone, no tensorflow needed)."""
    try:
        import urllib.request
        import gzip
        import os

        base_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
        files = {
            "train_images": "train-images-idx3-ubyte.gz",
            "train_labels": "train-labels-idx1-ubyte.gz",
            "test_images":  "t10k-images-idx3-ubyte.gz",
            "test_labels":  "t10k-labels-idx1-ubyte.gz",
        }

        cache_dir = Path.home() / ".mnist_cache"
        cache_dir.mkdir(exist_ok=True)

        def download(filename):
            path = cache_dir / filename
            if not path.exists():
                print(f"  Downloading {filename}...")
                urllib.request.urlretrieve(base_url + filename, path)
            return path

        def load_images(filename):
            with gzip.open(download(filename), "rb") as f:
                f.read(16)  # skip header
                return np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 784)

        def load_labels(filename):
            with gzip.open(download(filename), "rb") as f:
                f.read(8)   # skip header
                return np.frombuffer(f.read(), dtype=np.uint8)

        x_train = load_images("train-images-idx3-ubyte.gz").astype(np.float32) / 255.0
        y_train = load_labels("train-labels-idx1-ubyte.gz")
        x_test  = load_images("t10k-images-idx3-ubyte.gz").astype(np.float32) / 255.0
        y_test  = load_labels("t10k-labels-idx1-ubyte.gz")

        def onehot(y, n=10):
            out = np.zeros((len(y), n), dtype=np.float32)
            out[np.arange(len(y)), y] = 1.0
            return out

        return x_train, onehot(y_train), x_test, onehot(y_test), y_test

    except Exception as e:
        raise RuntimeError(f"Could not load MNIST: {e}")

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_deriv(s):
    return s * (1.0 - s)

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy(pred, true):
    return -np.mean(np.sum(true * np.log(pred + 1e-9), axis=1))

class GONeuralNet:
    """
    Two-layer fully-connected network with GO-constrained weights.
    Architecture: 784 -> hidden_size -> 10
    """
    def __init__(self, hidden_size, go_weight_levels, lr=0.05, seed=0):
        rng = np.random.default_rng(seed)
        self.lr     = lr
        self.levels = np.array(sorted(go_weight_levels))

        scale = 0.1
        self.W1 = rng.normal(0, scale, (784, hidden_size)).astype(np.float32)
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.W2 = rng.normal(0, scale, (hidden_size, 10)).astype(np.float32)
        self.b2 = np.zeros(10, dtype=np.float32)

        # Quantize at init — weights can only be GO states from the start
        self.W1 = quantize_to_go_states(self.W1, self.levels)
        self.W2 = quantize_to_go_states(self.W2, self.levels)

    def forward(self, x):
        self.x   = x
        self.z1  = x @ self.W1 + self.b1
        self.a1  = sigmoid(self.z1)
        self.z2  = self.a1 @ self.W2 + self.b2
        self.out = softmax(self.z2)
        return self.out

    def backward(self, y_true):
        m = y_true.shape[0]
        dz2 = (self.out - y_true) / m
        dW2 = self.a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * sigmoid_deriv(self.a1)
        dW1 = self.x.T @ dz1
        db1 = dz1.sum(axis=0)
        # Continuous gradient step
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        # Snap weights back to nearest GO state (straight-through estimator)
        self.W1 = quantize_to_go_states(self.W1, self.levels)
        self.W2 = quantize_to_go_states(self.W2, self.levels)

    def accuracy(self, x, y_labels):
        pred = np.argmax(self.forward(x), axis=1)
        return np.mean(pred == y_labels)

def train(net, x_train, y_train_oh, x_test, y_test_labels,
          epochs=50, batch_size=256):
    n = x_train.shape[0]
    loss_hist, acc_hist = [], []
    rng = np.random.default_rng(1)
    for epoch in range(epochs):
        idx = rng.permutation(n)
        x_train, y_train_oh = x_train[idx], y_train_oh[idx]
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            xb = x_train[start:start+batch_size]
            yb = y_train_oh[start:start+batch_size]
            out = net.forward(xb)
            epoch_loss += cross_entropy(out, yb)
            net.backward(yb)
        acc = net.accuracy(x_test, y_test_labels)
        loss_hist.append(epoch_loss / (n // batch_size))
        acc_hist.append(acc)
        print(f"  Epoch {epoch+1:2d}/{epochs}  loss={loss_hist[-1]:.4f}  "
              f"test_acc={acc*100:.1f}%")
    return loss_hist, acc_hist

# =============================================================================
# STEP 5: Accuracy vs number of GO states experiment
# =============================================================================

def accuracy_vs_states(x_train, y_train_oh, x_test, y_test_labels,
                       all_g_values, state_counts, hidden=64, epochs=10):
    results = {}
    for n in state_counts:
        print(f"\n--- {n} GO states ---")
        idx    = np.round(np.linspace(0, len(all_g_values)-1, n)).astype(int)
        subset = np.array(all_g_values)[idx]
        levels = conductance_to_weights(subset)
        centered = (levels * 2) - 1
        net    = GONeuralNet(hidden, centered, lr=0.5)
        _, acc_hist = train(net, x_train.copy(), y_train_oh.copy(),
                            x_test, y_test_labels, epochs=epochs)
        results[n] = acc_hist
    return results

# =============================================================================
# PLOTTING
# =============================================================================

def plot_all(curves, file_list, g_values, go_weight_levels,
             loss_hist, acc_hist, states_results=None):
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#0e0e0e")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    c = {"bg": "#0e0e0e", "fg": "#e0e0e0", "grid": "#2a2a2a",
         "accent": "#4fc3f7", "go": "#ffa726", "loss": "#ef5350", "acc": "#66bb6a"}

    def style(ax, title):
        ax.set_facecolor(c["bg"])
        ax.tick_params(colors=c["fg"], labelsize=8)
        ax.xaxis.label.set_color(c["fg"])
        ax.yaxis.label.set_color(c["fg"])
        ax.title.set_color(c["fg"])
        for spine in ax.spines.values():
            spine.set_edgecolor(c["grid"])
        ax.grid(True, color=c["grid"], linewidth=0.5)
        ax.set_title(title, fontsize=10, pad=8)

    # 1. I-V curves
    ax1 = fig.add_subplot(gs[0, 0])
    cmap = plt.cm.plasma
    for i, (v, curr) in enumerate(curves):
        color = cmap(i / max(len(curves)-1, 1))
        ax1.plot(v, curr * 1e9, color=color, linewidth=1, alpha=0.7)
    ax1.set_xlabel("Voltage (V)")
    ax1.set_ylabel("Current (nA)")
    style(ax1, f"I-V curves — {len(curves)} states")

    # 2. Conductance bar chart
    ax2 = fig.add_subplot(gs[0, 1])
    labels = [Path(f).stem for f in file_list]
    x_pos  = range(len(g_values))
    ax2.bar(x_pos, np.array(g_values) * 1e9,
            color=c["go"], edgecolor=c["bg"], linewidth=0.3)
    ax2.set_xlabel("File (reduction state)")
    ax2.set_ylabel("Conductance (nS)")
    ax2.set_xticks(list(x_pos)[::max(1, len(g_values)//10)])
    ax2.set_xticklabels(labels[::max(1, len(g_values)//10)],
                        rotation=45, ha="right", fontsize=7)
    style(ax2, "Extracted conductance per state")

    # 3. Weight levels
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(go_weight_levels, np.zeros_like(go_weight_levels),
                s=60, color=c["go"], zorder=5, edgecolors="white", linewidths=0.5)
    for w in go_weight_levels:
        ax3.axvline(w, color=c["go"], alpha=0.2, linewidth=0.8, linestyle="--")
    ax3.set_xlabel("Normalized weight value")
    ax3.set_yticks([])
    ax3.set_xlim(-0.05, 1.05)
    style(ax3, f"GO weight levels ({len(centered_levels)} discrete states)")

    # 4. Training loss
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(range(1, len(loss_hist)+1), loss_hist,
             color=c["loss"], linewidth=2, marker="o", markersize=4)
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Cross-entropy loss")
    style(ax4, "Training loss")

    # 5. Test accuracy
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(range(1, len(acc_hist)+1), [a*100 for a in acc_hist],
             color=c["acc"], linewidth=2, marker="o", markersize=4)
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("Test accuracy (%)")
    ax5.set_ylim(0, 100)
    style(ax5, f"MNIST test accuracy (final: {acc_hist[-1]*100:.1f}%)")

    # 6. Accuracy vs states
    ax6 = fig.add_subplot(gs[1, 2])
    if states_results:
        ns     = sorted(states_results.keys())
        finals = [states_results[n][-1] * 100 for n in ns]
        ax6.plot(ns, finals, color=c["accent"], linewidth=2, marker="o", markersize=6)
        ax6.set_xlabel("Number of GO states")
        ax6.set_ylabel("Final test accuracy (%)")
        ax6.set_xticks(ns)
        ax6.set_ylim(0, 100)
        style(ax6, "Accuracy vs GO state resolution")
    else:
        ax6.text(0.5, 0.5, "Set RUN_STATE_SWEEP = True\nto generate this plot",
                 transform=ax6.transAxes, ha="center", va="center",
                 color=c["fg"], fontsize=9)
        style(ax6, "Accuracy vs GO state resolution")

    plt.suptitle("Graphene Oxide Neural Network Simulation",
                 color=c["fg"], fontsize=13, y=1.01)
    plt.savefig("go_nn_results.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("\nPlot saved to go_nn_results.png")
    plt.show()

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print(f"Loading {len(IV_FILES)} I-V files...")
    curves = load_iv_files(IV_FILES)

    if len(curves) == 0:
        raise RuntimeError("No files loaded — check your IV_FILES list and that "
                           "you are running the script from the right folder.")

    print(f"Successfully loaded {len(curves)} files.\n")

    g_values = [extract_conductance(v, i, CONDUCTANCE_METHOD, EVAL_VOLTAGE)
                for v, i in curves]

    print("Extracted conductances (nS):")
    for fname, g in zip(IV_FILES[:len(g_values)], g_values):
        print(f"  {fname:<20} {g*1e9:.4f} nS")

    go_weight_levels = conductance_to_weights(g_values)
    print(f"\nNormalized weight levels ({len(go_weight_levels)} states):")
    print(np.round(go_weight_levels, 3))

    print("\nLoading MNIST...")
    x_train, y_train_oh, x_test, y_test_oh, y_test_labels = load_mnist()
    print(f"  Train: {x_train.shape}  Test: {x_test.shape}")

    print("\nTraining GO-constrained network on MNIST...")
    # Pick 12 evenly spaced states from your 68
    n_states = 12
    idx = np.round(np.linspace(0, len(go_weight_levels)-1, n_states)).astype(int)
    subset_levels = go_weight_levels[idx]
    centered_levels = (subset_levels * 2) - 1
    net = GONeuralNet(hidden_size=128, go_weight_levels=centered_levels, lr=0.5)
    loss_hist, acc_hist = train(net, x_train, y_train_oh,
                                    x_test, y_test_labels, epochs=50)

    states_results = None
    if RUN_STATE_SWEEP:
        print("\nRunning state sweep...")
        state_counts = list(range(2, 13))  # 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
        states_results = accuracy_vs_states(
            x_train, y_train_oh, x_test, y_test_labels,
            g_values, state_counts, hidden=64, epochs=10
        )

    plot_all(curves, IV_FILES[:len(curves)], g_values, go_weight_levels,
             loss_hist, acc_hist, states_results)