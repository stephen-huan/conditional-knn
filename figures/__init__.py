import matplotlib.pyplot as plt
import numpy as np

# display settings
np.set_printoptions(precision=3, suppress=True)

# set random seed
rng = np.random.default_rng(1)

### helper methods


def save_1d__(fname: str, args, root: str) -> None:
    """Write the 1D array to the file."""
    np.savetxt(f"{root}/data/{fname}", np.array(list(zip(*args))))


def save__(fname: str, num_rows: int, args, root: str) -> None:
    """Write the 2D array to the file."""
    with open(f"{root}/data/{fname}", "w") as f:
        for i, row in enumerate(list(zip(*args))):
            if i > 0 and i % num_rows == 0:
                f.write("\n")
            f.write(" ".join(map(str, row)) + "\n")


### graphing

lightblue = "#a1b4c7"
orange = "#ea8810"
silver = "#b0aba8"
rust = "#b8420f"
seagreen = "#23553c"

lightsilver = "#e7e6e5"
darkorange = "#c7740e"
darksilver = "#96918f"
darklightblue = "#8999a9"
darkrust = "#9c380d"
darkseagreen = "#1e4833"

plt.style.use("seaborn-paper")
