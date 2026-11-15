import os
import numpy as np

SEED = 42

def main():
    raw = np.load("data/mnist.npz")
    x_train = raw["x_train"].reshape(-1, 784).astype(np.float32) / 255.0
    y_train = raw["y_train"].astype(int)
    x_test = raw["x_test"].reshape(-1, 784).astype(np.float32) / 255.0
    y_test = raw["y_test"].astype(int)

    k = y_train != 0
    x_train, y_train = x_train[k], y_train[k]
    k = y_test != 0
    x_test, y_test = x_test[k], y_test[k]

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(x_train))
    val_idx, tr_idx = idx[:10000], idx[10000:]

    os.makedirs("data", exist_ok=True)
    np.savez_compressed(
        "data/mnist_1to9.npz",
        x_train=x_train[tr_idx], y_train=y_train[tr_idx],
        x_val=x_train[val_idx], y_val=y_train[val_idx],
        x_test=x_test, y_test=y_test,
    )
    print("train", len(tr_idx), "val", len(val_idx), "test", len(x_test))

if __name__ == "__main__":
    main()
