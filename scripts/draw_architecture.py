"""Draw the CIFAR-10 SwitchNet V1 architecture as a standalone SVG."""

from pathlib import Path
from xml.sax.saxutils import escape


OUTPUT = Path(__file__).resolve().parents[1] / "reports" / "switchnet_cifar10_architecture.svg"


def box(x, y, w, h, title, detail, fill="#eff6ff", stroke="#2563eb"):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        f'<text x="{x+w/2}" y="{y+30}" class="title" text-anchor="middle">{escape(title)}</text>'
        f'<text x="{x+w/2}" y="{y+53}" class="detail" text-anchor="middle">{escape(detail)}</text>'
    )


def line(x1, y1, x2, y2):
    return f'<path d="M{x1} {y1} L{x2} {y2}" stroke="#475569" stroke-width="2.5" fill="none" marker-end="url(#arrow)"/>'


parts = [
    '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="850" viewBox="0 0 1100 850">',
    '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path d="M0 0 L9 4.5 L0 9 Z" fill="#475569"/></marker></defs>',
    '<style>text{font-family:Arial,sans-serif;fill:#0f172a}.heading{font-size:28px;font-weight:700}.title{font-size:18px;font-weight:700}.detail{font-size:14px}.note{font-size:15px;fill:#475569}</style>',
    '<rect width="1100" height="850" fill="#f8fafc"/>',
    '<text x="550" y="45" class="heading" text-anchor="middle">SwitchNet V1 · CIFAR-10</text>',
    box(390, 75, 320, 65, "Ảnh đầu vào", "RGB · 3 × 32 × 32", "#f1f5f9", "#64748b"),
    line(550, 140, 550, 170),
    box(345, 175, 410, 65, "CNN encoder dùng chung", "Conv 3→32 → Conv 32→32 → MaxPool", "#dbeafe"),
    line(550, 240, 550, 260),
    box(345, 265, 410, 65, "Tiếp tục trích xuất đặc trưng", "Conv 32→64 → MaxPool → Conv 64→128", "#dbeafe"),
    line(550, 330, 550, 350),
    box(345, 355, 410, 65, "Gom và chiếu đặc trưng", "Global average pool → Linear 128→128", "#dbeafe"),
    line(550, 420, 550, 450),
    box(390, 455, 320, 65, "Vector đặc trưng z", "128 chiều · chia sẻ cho mọi expert", "#dcfce7", "#16a34a"),
    '<path d="M550 520 L550 552 M190 552 L910 552" stroke="#475569" stroke-width="2.5" fill="none"/>',
    line(190, 552, 190, 575), line(430, 552, 430, 575), line(670, 552, 670, 575), line(910, 552, 910, 575),
    box(75, 580, 230, 85, "Expert lớp 0", "Adapter 128→16→128; score 0", "#fef3c7", "#d97706"),
    box(315, 580, 230, 85, "Expert lớp 1", "Adapter 128→16→128; score 1", "#fef3c7", "#d97706"),
    box(555, 580, 230, 85, "… Expert lớp 2–8", "Cùng cấu trúc, tham số riêng", "#fef3c7", "#d97706"),
    box(795, 580, 230, 85, "Expert lớp 9", "Adapter 128→16→128; score 9", "#fef3c7", "#d97706"),
    '<path d="M190 665 L190 700 M430 665 L430 700 M670 665 L670 700 M910 665 L910 700 M190 700 L910 700" stroke="#475569" stroke-width="2.5" fill="none"/>',
    line(550, 700, 550, 720),
    box(330, 725, 440, 65, "Ghép 10 điểm số → dự đoán", "Logits [10] · chọn lớp có điểm cao nhất", "#e0e7ff", "#4f46e5"),
    '<text x="550" y="822" class="note" text-anchor="middle">Khi train: nhãn chỉ điều khiển gradient của expert; khi dự đoán: không cần nhãn.</text>',
    '</svg>',
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text("\n".join(parts), encoding="utf-8")
print(OUTPUT)


GATE_OUTPUT = OUTPUT.with_name("switchnet_gate_cifar10_architecture.svg")
gate = [
    '<svg xmlns="http://www.w3.org/2000/svg" width="1250" height="960" viewBox="0 0 1250 960">',
    '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path d="M0 0 L9 4.5 L0 9 Z" fill="#475569"/></marker></defs>',
    '<style>text{font-family:Arial,sans-serif;fill:#0f172a}.heading{font-size:28px;font-weight:700}.title{font-size:18px;font-weight:700}.detail{font-size:14px}.note{font-size:15px;fill:#475569}.small{font-size:13px;fill:#475569}</style>',
    '<rect width="1250" height="960" fill="#f8fafc"/>',
    '<text x="625" y="44" class="heading" text-anchor="middle">SwitchNet Gate · CIFAR-10</text>',
    box(455, 65, 340, 64, "Ảnh RGB", "3 × 32 × 32", "#f1f5f9", "#64748b"),
    line(625, 129, 625, 150),
    box(400, 155, 450, 70, "CNN encoder dùng chung", "Conv/BatchNorm/Pool → vector z, 128 chiều", "#dbeafe"),
    line(625, 225, 625, 249),
    box(420, 255, 410, 66, "10 expert theo lớp", "Mỗi expert: adapter residual + evidence score", "#fef3c7", "#d97706"),
    '<text x="625" y="347" class="small" text-anchor="middle">Mọi expert đều chạy forward cho mọi ảnh</text>',
    line(625, 321, 625, 353),
    '<path d="M625 359 L625 377 M290 377 L960 377" stroke="#475569" stroke-width="2.5" fill="none"/>',
    line(290, 377, 290, 395), line(960, 377, 960, 395),
    box(110, 400, 360, 70, "Đặc trưng expert", "h₀ … h₉ · mỗi vector 128 chiều", "#dcfce7", "#16a34a"),
    box(780, 400, 360, 70, "Điểm evidence", "s₀ … s₉ · 10 điểm số", "#ede9fe", "#7c3aed"),
    '<path d="M290 470 L290 500 L625 500 M960 470 L960 500 L625 500" stroke="#475569" stroke-width="2.5" fill="none"/>',
    line(625, 500, 625, 515),
    box(400, 520, 450, 70, "Gate học trọng số", "Concat(h₀…h₉, s₀…s₉) → MLP → softmax", "#fce7f3", "#db2777"),
    line(625, 590, 625, 610),
    box(400, 615, 450, 70, "Kết hợp đặc trưng", "fused = Σ trọng sốᵢ × hᵢ", "#fce7f3", "#db2777"),
    line(625, 685, 625, 705),
    box(400, 710, 450, 70, "Integration head", "Linear: 128 → 10 logits", "#fce7f3", "#db2777"),
    '<path d="M960 470 L1100 470 L1100 817 L820 817" stroke="#7c3aed" stroke-width="2.5" fill="none" marker-end="url(#arrow)"/>',
    '<text x="1100" y="640" class="small" text-anchor="middle">đường cộng</text>',
    '<text x="1100" y="660" class="small" text-anchor="middle">evidence trực tiếp</text>',
    line(625, 780, 625, 795),
    box(360, 800, 530, 70, "Logits cuối = head(fused) + scores", "Chọn lớp có logit lớn nhất", "#e0e7ff", "#4f46e5"),
    '<rect x="125" y="895" width="1000" height="42" rx="10" fill="#e2e8f0"/>',
    '<text x="625" y="922" class="note" text-anchor="middle">Train: chỉ expert đúng lớp nhận gradient ở tham số riêng; CNN chung và gate vẫn được cập nhật.</text>',
    '</svg>',
]
GATE_OUTPUT.write_text("\n".join(gate), encoding="utf-8")
print(GATE_OUTPUT)
