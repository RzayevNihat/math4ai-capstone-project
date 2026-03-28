import json
from plot_utils import plot_training_curves, plot_optimizer_summary


def main():
    with open("starter_pack/results/digits_optimizer_study.json", "r", encoding="utf-8") as f:
        results = json.load(f)

    plot_training_curves(results, save_dir="starter_pack/figures")
    plot_optimizer_summary(results, save_dir="starter_pack/figures")

    print("Plots saved in starter_pack/figures")


if __name__ == "__main__":
    main()