import re
import argparse

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="绘制 CycleGAN 训练损失曲线")
    parser.add_argument("--log_path", type=str, required=True, help="loss_log.txt 路径")
    parser.add_argument("--out", type=str, default=None, help="输出图片路径（默认与日志同目录下的 loss_curve.png）")
    return parser.parse_args()


def main():
    args = parse_args()

    # Auto-detect all loss names from the first matching line
    loss_dict = {}
    loss_order = []
    pattern = re.compile(r"([A-Za-z_]+): ([0-9.]+)")

    with open(args.log_path, "r") as f:
        for line in f:
            found = pattern.findall(line)
            for k, v in found:
                if k not in loss_dict:
                    loss_dict[k] = []
                    loss_order.append(k)
                loss_dict[k].append(float(v))

    if not loss_dict:
        print("未在日志中找到任何损失值。")
        return

    print(f"检测到损失项: {loss_order}")
    print(f"每项有 {len(loss_dict[loss_order[0]])} 个记录点")

    # Split into subplots: G-loss / D-loss / cycle & edge loss
    g_keys = [k for k in loss_order if k.startswith("G_")]
    d_keys = [k for k in loss_order if k.startswith("D_")]
    aux_keys = [k for k in loss_order if k not in g_keys and k not in d_keys]

    n_sub = sum(bool(x) for x in [g_keys, d_keys, aux_keys])
    fig, axes = plt.subplots(n_sub, 1, figsize=(10, 4 * n_sub), sharex=True)
    if n_sub == 1:
        axes = [axes]

    row = 0
    for keys, title in [(g_keys, "Generator Loss"), (d_keys, "Discriminator Loss"), (aux_keys, "Cycle / Identity / Edge Loss")]:
        if not keys:
            continue
        ax = axes[row]
        for k in keys:
            ys = loss_dict[k]
            if len(ys) > 500:
                # Downsample for cleaner plot
                step = max(1, len(ys) // 500)
                idx = list(range(0, len(ys), step))
                ax.plot(idx, [ys[i] for i in idx], alpha=0.8, linewidth=0.8, label=k)
            else:
                ax.plot(ys, alpha=0.8, linewidth=0.8, label=k)
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)
        row += 1

    axes[-1].set_xlabel("Iteration")

    out_path = args.out or args.log_path.replace("loss_log.txt", "loss_curve.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
