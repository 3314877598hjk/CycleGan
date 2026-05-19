# 基于改进 CycleGAN 的无配对图像迁移研究与可视化系统

> **毕业设计** · 面向地图矢量化与风格迁移任务的 CycleGAN 改进研究与系统实现

---

## 项目简介

本项目以**无配对图像迁移**为研究背景，在经典 CycleGAN 框架上进行了以下改进：

1. **自注意力增强生成器（Self-Attention Generator）**：在 ResNet 生成器残差块之后插入轻量自注意力模块，使模型在全局范围内建立空间依赖，提升对道路、边界、区域块等关键结构的捕捉能力。

2. **Sobel 边缘一致性损失（Edge Consistency Loss）**：针对地图风格迁移任务中轮廓易失真的问题，设计了基于 Sobel 算子的结构保持损失，约束生成图像的边缘分布与输入一致。

3. **多任务推理系统与显存调度（Multi-Task Demo with LRU Scheduling）**：实现了基于 Gradio 的在线推理界面，支持多任务切换；采用 LRU 缓存策略限制同时驻留显存中的模型数量，并在任务切换时自动将旧模型迁移至 CPU、释放显存。

### 主要任务场景

| 任务 | 方向 | 说明 |
|------|------|------|
| Map → 地图渲染风格图 | A → B | 卫星/地图图像 → 渲染风格地图 |
| 地图渲染风格图 → Map | B → A | 渲染风格地图 → 卫星/地图图像 |
| Horse → Zebra | A → B | 马 → 斑马（验证泛化能力） |
| Zebra → Horse | B → A | 斑马 → 马 |

---

## 消融实验设计

本项目设计了四组对照实验，用于验证各改进点的有效性：

| 组别 | 自注意力 | 边缘损失 | 说明 |
|------|---------|---------|------|
| **G0** | ✗ | ✗ | 原始 CycleGAN 基线 |
| **G1** | ✗ | ✓ | + 边缘一致性损失 |
| **G2** | ✓ | ✗ | + 自注意力模块 |
| **G3** | ✓ | ✓ | 完整改进模型 |

---

## 项目结构

```
pytorch-CycleGAN-and-pix2pix/
├── models/
│   ├── networks.py          # 网络结构：SelfAttention、EdgeLoss、ResnetGenerator
│   ├── cycle_gan_model.py   # CycleGAN 模型：支持边缘损失和注意力开关
│   └── ...
├── scripts/
│   ├── eval_metrics.py      # 评估脚本（SSIM / LPIPS / FID / 边缘保持率）
│   └── benchmark_inference.py  # 推理性能基准（延迟 / 显存 / 吞吐量）
├── app.py                   # Gradio 多任务演示系统（LRU 显存调度）
├── plot_loss.py             # 训练损失可视化
├── train.py                 # 训练入口（支持 DDP 多卡）
├── test.py                  # 推理/测试入口
└── checkpoints/             # 预训练模型与训练日志
```

---

## 环境安装

```bash
# 创建 conda 环境
conda create -n cyclegan python=3.11
conda activate cyclegan

# 安装 PyTorch（根据实际 CUDA 版本调整）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装项目依赖
pip install -r requirements.txt
```

---

## 训练

### G0：原始 CycleGAN 基线

```bash
python train.py \
    --dataroot ./datasets/maps \
    --name G0_baseline \
    --model cycle_gan \
    --no_attention \
    --n_epochs 100 --n_epochs_decay 100
```

### G1：+ 边缘一致性损失

```bash
python train.py \
    --dataroot ./datasets/maps \
    --name G1_edge_loss \
    --model cycle_gan \
    --no_attention \
    --use_edge_loss --lambda_edge 1.0 \
    --n_epochs 100 --n_epochs_decay 100
```

### G2：+ 自注意力模块

```bash
python train.py \
    --dataroot ./datasets/maps \
    --name G2_attention \
    --model cycle_gan \
    --n_epochs 100 --n_epochs_decay 100
```

### G3：完整改进模型（默认）

```bash
python train.py \
    --dataroot ./datasets/maps \
    --name G3_full \
    --model cycle_gan \
    --use_edge_loss --lambda_edge 1.0 \
    --n_epochs 100 --n_epochs_decay 100
```

### 多卡训练（DDP）

```bash
torchrun --nproc_per_node=2 train.py \
    --dataroot ./datasets/maps \
    --name G3_full \
    --model cycle_gan \
    --use_edge_loss
```

---

## 测试与评估

### 生成结果

```bash
python test.py \
    --dataroot ./datasets/maps \
    --name G3_full \
    --model cycle_gan \
    --direction AtoB
```

### 量化评估（SSIM / LPIPS / 边缘保持率）

```bash
# 评估 G0 基线
python scripts/eval_metrics.py \
    --real_dir datasets/maps/testB \
    --fake_dir results/G0_baseline/test_latest/images/fake_B

# 评估 G3 完整模型，并计算 FID
python scripts/eval_metrics.py \
    --real_dir datasets/maps/testB \
    --fake_dir results/G3_full/test_latest/images/fake_B \
    --compute_fid
```

### 系统性能测试

```bash
# 单任务推理延迟与显存
python scripts/benchmark_inference.py --task "Map -> Vector" --n_runs 50

# 多任务切换 LRU 缓存效果
python scripts/benchmark_inference.py --benchmark_switch

# 吞吐量测试
python scripts/benchmark_inference.py --benchmark_throughput --n_images 100
```

### 损失曲线可视化

```bash
python plot_loss.py --log_path checkpoints/G3_full/loss_log.txt
```

---

## 演示系统

```bash
python app.py
```

启动后在浏览器访问 `http://127.0.0.1:7860`，支持图像上传与任务选择。

---

## 核心改进说明

### 1. 自注意力模块

位于 [models/networks.py](models/networks.py)，`SelfAttention` 类。

将经典的 Query / Key / Value 注意力机制嵌入 ResNet 生成器，插入位置在全部残差块之后、上采样之前。注意力权重 γ 初始化为 0，使模块在训练初期等价于恒等映射，随训练推进逐步学习全局关联。

训练时通过 `--no_attention` 关闭（用于 G0/G1 消融基线），默认开启。

### 2. Sobel 边缘一致性损失

位于 [models/networks.py](models/networks.py)，`EdgeLoss` 类；集成于 [models/cycle_gan_model.py](models/cycle_gan_model.py)。

对输入图像和生成图像分别提取灰度 Sobel 边缘图，计算 L1 损失：

```
L_edge = λ_edge × (L1(edge(fake_B), edge(real_A)) + L1(edge(fake_A), edge(real_B)))
```

对地图矢量化任务，该损失直接约束道路边界和区域轮廓的保留。通过 `--use_edge_loss` 开启，`--lambda_edge` 控制权重（默认 1.0）。

### 3. LRU 显存调度

位于 [app.py](app.py)，`LRUModelCache` 类。

当用户切换任务时：
1. 将 LRU 最旧模型的所有子网络迁移至 CPU
2. 调用 `gc.collect()` 释放 Python 对象
3. 调用 `torch.cuda.empty_cache()` 释放 CUDA 缓存
4. 加载新模型至 GPU

---

## 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--no_attention` | 关闭（默认开启注意力） | 禁用自注意力模块 |
| `--use_edge_loss` | False | 开启边缘一致性损失 |
| `--lambda_edge` | 1.0 | 边缘损失权重 |
| `--lambda_A` | 10.0 | A→B→A 循环一致损失权重 |
| `--lambda_B` | 10.0 | B→A→B 循环一致损失权重 |
| `--lambda_identity` | 0.5 | 身份映射损失权重 |
| `--gan_mode` | lsgan | GAN 损失类型（lsgan / vanilla / wgangp） |
| `--netG` | resnet_9blocks | 生成器类型 |
| `--norm` | instance | 归一化层类型 |
| `--lr` | 0.0002 | Adam 学习率 |
| `--n_epochs` | 100 | 恒定学习率训练轮数 |
| `--n_epochs_decay` | 100 | 学习率线性衰减轮数 |

---

## 参考文献

本项目基于以下工作：

- **CycleGAN**: Zhu et al., *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*, ICCV 2017. [[arxiv]](https://arxiv.org/abs/1703.10593)
- **Self-Attention GAN**: Zhang et al., *Self-Attention Generative Adversarial Networks*, ICML 2019.
- **PatchGAN / pix2pix**: Isola et al., *Image-to-Image Translation with Conditional Adversarial Networks*, CVPR 2017.

原始框架代码来源：[junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
