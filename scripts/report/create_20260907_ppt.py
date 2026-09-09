from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt


OUT = Path("docs/report/20260907.pptx")
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

BG = RGBColor(247, 246, 241)
INK = RGBColor(31, 48, 42)
GREEN = RGBColor(46, 106, 75)
MINT = RGBColor(220, 235, 222)
SAGE = RGBColor(194, 216, 198)
GOLD = RGBColor(207, 154, 57)
GOLD_PALE = RGBColor(246, 235, 206)
LINE = RGBColor(210, 216, 207)
MUTED = RGBColor(92, 105, 96)
WHITE = RGBColor(255, 255, 255)
FONT = "Microsoft YaHei"


def add_rect(slide, x, y, w, h, fill, line=None, radius=False):
    typ = MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE if radius else MSO_AUTO_SHAPE_TYPE.RECTANGLE
    shape = slide.shapes.add_shape(typ, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line if line else fill
    if radius:
        shape.adjustments[0] = 0.1
    return shape


def add_text(slide, text, x, y, w, h, size=18, color=INK, bold=False,
             align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP, margin=0.03):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(margin)
    tf.vertical_anchor = valign
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return box


def add_title(slide, number, title, subtitle=None):
    add_rect(slide, 0, 0, 13.333, 0.14, GREEN)
    add_text(slide, f"{number:02d}", 0.63, 0.38, 0.55, 0.35, 13, GREEN, True)
    add_text(slide, title, 1.16, 0.29, 11.4, 0.53, 25, INK, True)
    if subtitle:
        add_text(slide, subtitle, 1.18, 0.84, 11.2, 0.28, 10.5, MUTED)
    add_rect(slide, 0.65, 1.2, 12.03, 0.012, LINE)


def footer(slide, number):
    add_text(slide, "农业智能识别项目｜双周进展", 0.65, 7.14, 3.7, 0.18, 8.5, MUTED)
    add_text(slide, str(number), 12.25, 7.14, 0.4, 0.18, 8.5, MUTED, align=PP_ALIGN.RIGHT)


def card(slide, x, y, w, h, label, value, note=None, accent=GREEN):
    add_rect(slide, x, y, w, h, WHITE, LINE, radius=True)
    add_rect(slide, x, y, 0.07, h, accent, accent, radius=True)
    add_text(slide, label, x + 0.25, y + 0.2, w - 0.42, 0.28, 10, MUTED, True)
    add_text(slide, value, x + 0.25, y + 0.54, w - 0.42, 0.5, 25, accent, True)
    if note:
        add_text(slide, note, x + 0.25, y + 1.12, w - 0.42, 0.24, 9, MUTED)


def table(slide, rows, x, y, w, h, widths=None, font_size=10, highlights=None):
    nrows, ncols = len(rows), len(rows[0])
    shape = slide.shapes.add_table(nrows, ncols, Inches(x), Inches(y), Inches(w), Inches(h))
    tbl = shape.table
    if widths:
        for col, width in zip(tbl.columns, widths):
            col.width = Inches(width)
    highlights = highlights or set()
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            cell = tbl.cell(r, c)
            cell.margin_left = cell.margin_right = Inches(0.05)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.fill.solid()
            cell.fill.fore_color.rgb = GREEN if r == 0 else (MINT if (r, c) in highlights else WHITE)
            cell.border if False else None
            tf = cell.text_frame
            tf.clear()
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if c else PP_ALIGN.LEFT
            run = p.add_run()
            run.text = str(value)
            run.font.name = FONT
            run.font.size = Pt(font_size)
            run.font.bold = r == 0 or (r, c) in highlights
            run.font.color.rgb = WHITE if r == 0 else INK
    return shape


def bullet(slide, text, x, y, w, h=0.38, size=14, dot_color=GREEN):
    add_rect(slide, x, y + 0.13, 0.08, 0.08, dot_color, dot_color, radius=True)
    add_text(slide, text, x + 0.2, y, w - 0.2, h, size, INK)


def new_slide(number, title, subtitle=None):
    slide = prs.slides.add_slide(BLANK)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = BG
    add_title(slide, number, title, subtitle)
    footer(slide, number)
    return slide


# 01 Motivation
slide = new_slide(1, "从直接识别到分类器 + HCV", "分别处理已知类与未知类问题")
add_text(slide, "核心想法", 0.78, 1.53, 2.2, 0.35, 14, GREEN, True)
add_text(slide, "已知类主要靠分类模型做稳；未知类和不确定样本再通过 HCV / RAG 处理。", 0.78, 1.96, 5.2, 0.84, 21, INK, True)
add_text(slide, "不再把 Direct 和 RAG 当成互相替代的两条路线。", 0.78, 2.88, 5.15, 0.3, 13, MUTED)
steps = [
    ("图片", "输入图像"),
    ("分类模型", "Known 类候选"),
    ("LLM 判断", "图像与类别证据"),
    ("RAG 验证", "不确定 / Unknown"),
    ("最终类别", "输出结果"),
]
for i, (name, note) in enumerate(steps):
    x = 0.85 + i * 2.38
    fill = MINT if i in (1, 3) else WHITE
    accent = GREEN if i != 3 else GOLD
    add_rect(slide, x, 4.42, 1.78, 1.16, fill, LINE, radius=True)
    add_text(slide, name, x + 0.12, 4.66, 1.54, 0.25, 13, accent, True, align=PP_ALIGN.CENTER)
    add_text(slide, note, x + 0.12, 5.06, 1.54, 0.2, 9.5, MUTED, align=PP_ALIGN.CENTER)
    if i < len(steps) - 1:
        add_text(slide, "→", x + 1.82, 4.78, 0.42, 0.28, 20, GREEN, True, align=PP_ALIGN.CENTER)
add_text(slide, "Known", 3.28, 6.02, 1.0, 0.25, 10, GREEN, True, align=PP_ALIGN.CENTER)
add_text(slide, "Unknown / 不确定", 7.72, 6.02, 1.65, 0.25, 10, GOLD, True, align=PP_ALIGN.CENTER)


# 02 Data
slide = new_slide(2, "重新整理数据与划分", "重新划分 Known / Unknown，并将数据清洗和类别规范化前置")
bullets = [
    "按类别长尾程度和视觉近邻关系重新划分 Known / Unknown。",
    "使用人工审阅图像建立测试集，覆盖全部类别。",
    "统一类别名称，修复重复类别和不规范表述。",
    "完成 pHash 去重、数据约束清洗和不足类别补采。",
    "将部分失败样本改造成纠错样本，先检查图像与 ground truth 是否一致。",
]
for i, item in enumerate(bullets):
    bullet(slide, item, 0.78, 1.55 + i * 0.66, 5.65, 0.48, 12.2)
data_rows = [
    ["数据集 / 划分", "Overall", "Disease", "Pest"],
    ["类别数", "217（109 / 108）", "145（73 / 72）", "72（36 / 36）"],
    ["Dev", "812（542 / 270）", "542（362 / 180）", "270（180 / 90）"],
    ["Test", "1,019（534 / 485）", "547（283 / 264）", "472（251 / 221）"],
    ["数据库", "401（192 / 209）", "286（136 / 150）", "115（56 / 59）"],
    ["训练候选图像", "141,482（Known）", "69,907", "71,575"],
]
table(slide, data_rows, 6.72, 1.56, 5.93, 3.68, [1.42, 1.66, 1.45, 1.4], 9.3,
      {(3, 0), (3, 1), (3, 2), (3, 3)})
add_rect(slide, 6.72, 5.54, 5.93, 0.8, GOLD_PALE, GOLD_PALE, radius=True)
add_text(slide, "测试集 1,019 张，覆盖全部类别；后续结果同时报告 Known 和 Unknown。", 6.96, 5.78, 5.44, 0.24, 11.4, INK, True)


# 03 4B
slide = new_slide(3, "4B 模型：200 step 对照", "Direct-only 更偏向 Known 类；RAG 训练有助于保持 Unknown 泛化")
rows4 = [
    ["训练策略", "推理", "语言", "Overall", "Known", "Unknown", "Known−Unknown"],
    ["Base", "Direct", "EN", "2.55", "2.39", "2.74", "-0.35pp"],
    ["", "Direct", "ZH", "8.34", "8.62", "8.02", "0.60pp"],
    ["", "RAG", "EN", "16.98", "16.33", "17.72", "-1.39pp"],
    ["", "RAG", "ZH", "19.04", "17.61", "17.72", "-0.11pp"],
    ["Direct-Only", "Direct", "EN", "30.91", "46.24", "13.29", "32.95pp"],
    ["", "Direct", "ZH", "25.91", "38.53", "11.39", "27.14pp"],
    ["", "RAG", "EN", "30.23", "44.95", "13.29", "31.66pp"],
    ["", "RAG", "ZH", "25.32", "37.06", "11.81", "25.25pp"],
    ["Direct + RAG", "Direct", "EN", "22.47", "35.96", "6.96", "29.00pp"],
    ["", "Direct", "ZH", "21.49", "30.83", "10.76", "20.07pp"],
    ["Direct + RAG", "RAG", "EN", "33.37", "34.31", "32.28", "2.03pp"],
    ["", "RAG", "ZH", "31.89", "34.86", "28.48", "6.38pp"],
]
table(slide, rows4, 0.75, 1.46, 11.87, 3.92, [1.66, 1.12, 0.78, 1.22, 1.22, 1.28, 1.55], 8.7,
      {(11, 3), (11, 5), (11, 6), (12, 3), (12, 5), (12, 6)})
bullet(slide, "Direct-only 主要提升 Known 类，对 Unknown 类的泛化仍然较弱。", 0.92, 5.78, 11.2, 0.34, 11.4)
bullet(slide, "Direct + RAG 的 RAG 路径缩小了 Known / Unknown 差距。", 0.92, 6.18, 11.2, 0.34, 11.4)
bullet(slide, "固定 200 step 下，混合训练可能稀释 Direct 样本训练量；Option 数据也需要单独做比例对照。", 0.92, 6.58, 11.2, 0.34, 11.4)


# 04 8B
slide = new_slide(4, "8B 模型", "Direct×5 + RAG 保留 RAG 泛化")
rows8 = [
    ["训练策略", "推理", "语言", "Overall", "Known", "Unknown"],
    ["Base", "Direct", "EN", "5.20", "5.32", "5.06"],
    ["", "Direct", "ZH", "9.03", "10.28", "7.59"],
    ["", "RAG", "EN", "21.69", "19.82", "23.84"],
    ["", "RAG", "ZH", "22.67", "20.00", "25.74"],
    ["Direct×5 + RAG", "Direct", "EN", "25.61", "37.98", "11.39"],
    ["", "Direct", "ZH", "17.96", "26.79", "7.81"],
    ["Direct×5 + RAG", "RAG", "EN", "33.86", "35.96", "31.43"],
    ["", "RAG", "ZH", "31.31", "29.36", "33.54"],
]
table(slide, rows8, 0.8, 1.55, 8.12, 3.18, [2.18, 1.25, 0.86, 1.22, 1.22, 1.38], 10,
      {(7, 3), (7, 5), (8, 3), (8, 5)})
add_rect(slide, 9.3, 1.55, 3.3, 3.18, MINT, MINT, radius=True)
add_text(slide, "当前观察", 9.62, 1.88, 2.6, 0.3, 14, GREEN, True)
add_text(slide, "RAG 训练后，Unknown 类表现没有像 Direct-only 一样明显下降。", 9.62, 2.4, 2.6, 0.85, 14, INK, True)
add_text(slide, "后续继续比较 Direct / RAG 的训练配比。", 9.62, 3.58, 2.58, 0.45, 11.4, MUTED)
add_rect(slide, 0.8, 5.22, 11.8, 1.2, WHITE, LINE, radius=True)
add_text(slide, "分类模型正在训练，尚未接入 HCV。后续用于 Known 类候选生成，再与 RAG 配合处理证据不足、候选冲突和 Unknown 样本。", 1.08, 5.58, 11.2, 0.36, 13, INK, True, align=PP_ALIGN.CENTER)


# 05 MAE + classifier
slide = new_slide(5, "MAE 与分类模型进展", "MAE 预训练完成，分类模型进入阶段性评测")
add_rect(slide, 0.78, 1.48, 3.72, 1.45, MINT, MINT, radius=True)
add_text(slide, "MAE 预训练", 1.06, 1.75, 2.2, 0.25, 12, GREEN, True)
add_text(slide, "1,668,771", 1.06, 2.07, 2.35, 0.38, 24, GREEN, True)
add_text(slide, "张农业图像，ViT-L/16 编码器", 1.06, 2.52, 2.95, 0.2, 9.8, MUTED)
add_rect(slide, 4.8, 1.48, 3.72, 1.45, WHITE, LINE, radius=True)
add_text(slide, "当前分类器", 5.08, 1.75, 2.2, 0.25, 12, GREEN, True)
add_text(slide, "训练中", 5.08, 2.07, 2.35, 0.38, 24, GREEN, True)
add_text(slide, "107 个 Known 类；暂未接入 HCV", 5.08, 2.52, 3.05, 0.2, 9.8, MUTED)
add_rect(slide, 8.82, 1.48, 3.72, 1.45, GOLD_PALE, GOLD_PALE, radius=True)
add_text(slide, "口径对齐", 9.1, 1.75, 2.2, 0.25, 12, GOLD, True)
add_text(slide, "107 → 109", 9.1, 2.07, 2.75, 0.38, 24, GOLD, True)
add_text(slide, "需与新 Known 类划分完成对齐", 9.1, 2.52, 3.05, 0.2, 9.8, MUTED)
metric_rows = [
    ["阶段性 Known-dev 结果", "Top-1", "Balanced Acc.", "Macro-F1", "Disease F1", "Pest F1"],
    ["当前最佳 checkpoint", "92.14%", "91.96%", "92.02%", "94.90%", "86.10%"],
]
table(slide, metric_rows, 0.78, 3.58, 11.76, 1.25, [3.15, 1.55, 1.9, 1.55, 1.8, 1.81], 12,
      {(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)})
add_text(slide, "当前结果只用于选择 checkpoint，不是最终测试结果。训练结束后先检查虫害和视觉近邻类别的混淆，再决定是否接入 HCV。", 0.92, 5.35, 11.45, 0.36, 12, INK, True, align=PP_ALIGN.CENTER)
add_text(slide, "Known-dev｜用于 checkpoint 选择｜非最终测试", 0.92, 5.92, 11.45, 0.22, 10.2, MUTED, align=PP_ALIGN.CENTER)


# 06 plan
slide = new_slide(6, "下周计划", "跑完分类模型，接入 HCV，再把训练配比拆开做对照")
plans = [
    ("01", "完成分类模型训练与评测", "查看类别级混淆，重点看虫害和视觉近邻类别；完成模型与类别口径对齐。"),
    ("02", "接入 HCV 推理", "先取 Known 类候选，再核对图像和证据；候选不可信或 Unknown 时走 RAG。"),
    ("03", "拆开训练配比做对照", "固定 200 step，比对 Direct、RAG、Option 数据比例，确认 Direct 能力下降的原因。"),
    ("04", "继续数据补齐与纠错", "补齐清洗后不足类别，完成纠错样本的教师一致性检查。"),
]
for i, (num, title, desc) in enumerate(plans):
    x = 0.78 + (i % 2) * 6.05
    y = 1.55 + (i // 2) * 2.23
    add_rect(slide, x, y, 5.7, 1.68, WHITE, LINE, radius=True)
    add_text(slide, num, x + 0.28, y + 0.3, 0.52, 0.3, 13, GREEN, True)
    add_text(slide, title, x + 0.94, y + 0.25, 4.38, 0.3, 15, INK, True)
    add_text(slide, desc, x + 0.94, y + 0.78, 4.38, 0.53, 11.2, MUTED)
add_rect(slide, 0.78, 6.18, 11.75, 0.52, MINT, MINT, radius=True)
add_text(slide, "需要回答：分类模型能否提高 Known 类效果；RAG 能否保持 Unknown 泛化；Direct、RAG、Option 的训练比例如何设置。", 1.02, 6.34, 11.25, 0.18, 10.7, INK, True, align=PP_ALIGN.CENTER)


OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print(OUT)
