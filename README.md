# SwitchNet: class-conditional private-gradient isolation

Đây là implementation nghiên cứu bằng PyTorch cho giả thuyết:

> Ground-truth label phân chia **quyền sở hữu gradient** trong các class-private
> adapter/normalization subspaces, trong khi tầng evidence readout của mỗi expert học
> từ cả mẫu dương và mẫu âm. Mọi expert vẫn cùng forward và cung cấp evidence tại inference.

Project cố ý không có confidence head trong core model. Nhãn chỉ được truyền vào forward
khi training/gradient analysis để điều khiển Jacobian; validation và inference luôn gọi
model không có nhãn.

## Các model/ablation

| Tên CLI | Nhiều expert | Private-gradient isolation | Specialization loss | Learned gate |
|---|---:|---:|---:|---:|
| `dense` | Không | Không | Không | Không |
| `multi_branch` | Có | Không | Không | Không |
| `switchnet_v1` | Có | Có | Không | Không |
| `switchnet_v2` | Có | Có | Có | Không |
| `switchnet_gate` | Có | Có | Có | Có |

`dense` dùng cùng encoder và tự chọn chiều rộng classifier sao cho tổng số parameter gần
`switchnet_v1` nhất. Metadata đối sánh được ghi trong `summary.json`. Mỗi expert là residual
bottleneck adapter nhẹ, vì vậy CIFAR-100 không nhân bản 100 encoder.

Ở V1/V2, vector evidence của tất cả expert chính là classification logits. Bản gate dùng
toàn bộ expert features và evidence để học trọng số, sau đó cộng direct evidence residual.
Mặc định gate tự tìm bottleneck/hidden width để tổng số parameter ngoài encoder gần V1,
tránh biến ablation thành so sánh capacity. Có thể tắt bằng `model.gate_parameter_match=false`.

## Bất biến gradient

Với expert `c` và mask theo từng mẫu `m_c = 1[y=c]`, code dùng hai đường adapter có cùng giá trị:

```python
r_trainable = adapter_c(z, parameters)
r_frozen = adapter_c(z, stop_gradient(parameters))
r = r_trainable * m_c + r_frozen * (1 - m_c)
score_c = evidence_c(r)  # readout học từ mọi mẫu
```

Hai đường tái sử dụng cùng dropout mask. Giá trị forward không đổi và private Jacobian của
adapter không phải target bằng 0, nhưng `r_frozen` vẫn truyền gradient theo `z` về shared
encoder. Tầng evidence readout nhận gradient từ cả mẫu dương và âm để học cách so sánh
điểm giữa các lớp; nó thuộc không gian integration, không thuộc không gian private.
Vì gradient của hai class khác nhau có support ở hai block private khác nhau, inner product
của chúng trong concatenated private parameter space bằng 0. Điều này **không** ngụ ý
gradient conflict trong shared encoder bằng 0; project đo hai không gian riêng.

## Cài đặt

Khuyến nghị Python 3.10–3.12 và một bản PyTorch phù hợp CUDA của máy:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,logging,flops]"
```

Nếu đã có môi trường PyTorch, chỉ cần `pip install -e .`. `tensorboard` và `thop` là tùy
chọn; thiếu chúng thì CSV logging vẫn chạy và FLOPs được ghi là `null`.

## Chạy nhanh

Một model trên synthetic Gaussian:

```bash
python -m switchnet.train --config configs/synthetic.yaml --model switchnet_v1
```

Toàn bộ ablation với cùng data split/seed:

```bash
python -m switchnet.train --config configs/synthetic.yaml --model all --run-name ablation_seed42
```

Smoke run nhỏ trên CPU:

```bash
python -m switchnet.train --config configs/synthetic.yaml --model all --run-name smoke \
  --set train.device=cpu --set train.epochs=1 --set data.train_size=192 \
  --set data.val_size=64 --set data.test_size=64 --set analysis.gradient_batches=1 \
  --set logging.tensorboard=false --set analysis.flops=false
```

Các override nhận cú pháp `section.key=value` và parse value như YAML.

## Binary, multi-class, overlap và imbalance

Binary synthetic:

```bash
python -m switchnet.train --config configs/synthetic.yaml --model all \
  --set data.num_classes=2
```

Điều khiển overlap bằng tỉ số `class_separation / noise_std`; separation càng nhỏ thì
overlap càng lớn. Long-tail dùng `imbalance_ratio = N_max / N_min` và phân bố exponential:

```bash
python -m switchnet.train --config configs/synthetic.yaml --model all \
  --set data.class_separation=0.75 --set data.imbalance_ratio=50
```

Validation/test synthetic luôn cân bằng để macro metrics không bị thay đổi chỉ bởi class
prior của training set.

## CIFAR-10 / CIFAR-100 và long-tail

Lần đầu tải dữ liệu, bật download rõ ràng:

```bash
python -m switchnet.train --config configs/cifar10.yaml --model all --set data.download=true
python -m switchnet.train --config configs/cifar100.yaml --model switchnet_v1 --set data.download=true
```

Sau đó có thể bỏ `data.download=true`. Ví dụ CIFAR-10-LT imbalance ratio 100:

```bash
python -m switchnet.train --config configs/cifar10.yaml --model all \
  --set data.imbalance_ratio=100
```

## Sweep

`scripts/run_sweep.py` chạy Cartesian product, bao gồm model ablation:

```bash
python scripts/run_sweep.py --base configs/synthetic.yaml --sweep configs/sweep_overlap.yaml
```

Kiểm tra commands trước khi chạy:

```bash
python scripts/run_sweep.py --base configs/synthetic.yaml \
  --sweep configs/sweep_overlap.yaml --dry-run
```

## Outputs và metrics

Mỗi run ghi:

- `config.resolved.yaml`: cấu hình chính xác;
- `metrics.csv` và tùy chọn TensorBoard events;
- `checkpoints/best.pt` cùng periodic checkpoints có optimizer/scheduler/RNG state;
- `summary.json`: accuracy, macro-F1, balanced accuracy, per-class recall/F1,
  confusion matrix, parameter count, FLOPs (nếu có), train/test time;
- `analysis/gradient_analysis.json` và CSV matrices cho shared/private cosine;
- `analysis/specialization.json`: score/adapter-residual matrix theo true class,
  selectivity, one-vs-rest AUROC, positive/negative score, calibration error và routing.

Mỗi experiment group có `combined_summary.csv`. Tổng hợp nhiều runs và tạo plot:

```bash
python scripts/summarize_results.py --runs runs --output reports
```

Script chỉ gộp các run có cùng resolved config (trừ model, seed, device và đường dẫn dữ
liệu), loại bản chạy trùng model/config/seed và ghi mỗi comparison vào thư mục fingerprint
riêng. Biểu đồ SVG không phụ thuộc GUI/backend plotting, nên chạy ổn trên server headless.
Smoke run, dataset, overlap và imbalance khác nhau không bị trộn chung.

TensorBoard:

```bash
tensorboard --logdir runs
```

## Resume

```bash
python -m switchnet.train --config configs/synthetic.yaml --model switchnet_v2 \
  --resume runs/synthetic/<run>/switchnet_v2/seed_42/checkpoints/epoch_0010.pt
```

Giữ nguyên các hyperparameter kiến trúc khi resume.
Checkpoint của các biến thể có isolation tạo trước bản sửa evidence readout không thể resume;
hãy bắt đầu run mới để tránh trộn hai quy tắc gradient khác nhau.
Checkpoint lưu cả Python/NumPy/PyTorch/CUDA RNG và generator state của DataLoader, nên thứ
tự batch/augmentation sau resume tiếp tục từ đúng trạng thái đã lưu.

## Tests

```bash
pytest
```

Hoặc không cần pytest:

```bash
python -m unittest discover -s tests -v
```

Các test quan trọng xác nhận:

1. non-target private adapter/normalization gradient bằng 0/`None`, còn evidence readout
   nhận gradient từ mẫu âm;
2. inner product giữa private gradient của hai class khác nhau bằng đúng 0;
3. mọi expert vẫn được gọi và label không thay đổi giá trị forward;
4. multi-branch control thật sự có non-target gradients;
5. binary/multi-class shapes, long-tail generation và parameter matching.
6. shared-encoder gradient được bảo toàn khi private expert gradient bị cô lập.

## Cấu trúc

```text
src/switchnet/
  models.py       encoders, experts, five model variants
  data.py         synthetic, CIFAR, deterministic long-tail
  engine.py       train/eval, AMP, checkpoint, reporting
  analysis.py     gradient conflict, specialization, FLOPs
  metrics.py      accuracy, macro-F1, per-class recall/F1
configs/          ready-to-run experiment configurations
scripts/          sweep and aggregation/plotting
tests/            gradient invariants and smoke tests
```

## Diễn giải khoa học đúng phạm vi

Architecture đảm bảo private-gradient overlap giữa class khác nhau bằng 0. Specialization,
accuracy và generalization **không** được đảm bảo; đó là các giả thuyết thực nghiệm. So sánh
`dense → multi_branch → V1 → V2 → gate`, sweep overlap, và long-tail giúp tách ảnh hưởng của
parameter count, branching, isolation, auxiliary objective và learned integration.
