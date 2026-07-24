"""测量结果导出工具。"""

import csv
import io
import json
import os
import tempfile
import urllib.error
import urllib.request
from functools import lru_cache

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
from PIL import Image


PHYSICS_LABELS = {
    "mode": "计算模式",
    "wavelength_nm": "波长 (nm)",
    "screen_distance_m": "屏距 (m)",
    "slit_spacing_mm": "双缝间距 (mm)",
    "spacing_mm": "实验间距 (mm)",
    "theoretical_spacing_mm": "理论间距 (mm)",
    "measured_spacing_mm": "测量间距 (mm)",
    "absolute_error_mm": "绝对误差 (mm)",
    "relative_error_percent": "相对误差 (%)",
}

MODE_LABELS = {
    "theoretical_spacing": "计算理论条纹间距",
    "wavelength": "反算波长",
    "slit_spacing": "反算双缝间距",
    "screen_distance": "反算屏距",
}

NOTO_CJK_URL = (
    "https://raw.githubusercontent.com/notofonts/noto-cjk/main/"
    "Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"
)


@lru_cache(maxsize=1)
def _chinese_font():
    """返回报告字体以及该字体是否支持中文。"""
    candidates = (
        "Microsoft YaHei",
        "Microsoft YaHei UI",
        "SimHei",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Source Han Sans CN",
        "WenQuanYi Zen Hei",
        "AR PL UMing CN",
        "Arial Unicode MS",
    )
    installed = {
        font.name: font.fname for font in font_manager.fontManager.ttflist
    }
    for name in candidates:
        if name in installed:
            return font_manager.FontProperties(fname=installed[name]), True

    font_path = os.path.join(tempfile.gettempdir(), "NotoSansCJKsc-Regular.otf")
    try:
        if not os.path.exists(font_path) or os.path.getsize(font_path) < 1_000_000:
            with urllib.request.urlopen(NOTO_CJK_URL, timeout=30) as response:
                font_data = response.read()
            if len(font_data) < 1_000_000 or font_data[:4] != b"OTTO":
                raise ValueError("下载的中文字体文件无效")
            with open(font_path, "wb") as font_file:
                font_file.write(font_data)
        font_manager.fontManager.addfont(font_path)
        return font_manager.FontProperties(fname=font_path), True
    except (OSError, ValueError, urllib.error.URLError):
        return font_manager.FontProperties(), False


def configure_matplotlib_chinese():
    """配置 Matplotlib 中文字体，并返回中文是否可用。"""
    report_font, supports_chinese = _chinese_font()
    plt.rcParams["axes.unicode_minus"] = False
    if supports_chinese:
        plt.rcParams["font.family"] = [report_font.get_name()]
    else:
        plt.rcParams["font.family"] = ["DejaVu Sans"]
    return supports_chinese


def _physics_payload_cn(physics_result):
    """将物理计算结果转换为适合留档的中文字段。"""
    if not physics_result:
        return None
    payload = {}
    for key, value in physics_result.items():
        label = PHYSICS_LABELS.get(key, key)
        if key == "mode":
            value = MODE_LABELS.get(value, value)
        payload[label] = value
    return payload


def result_csv_bytes(result):
    """生成每条条纹及相邻间距的 CSV。"""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "条纹序号", "峰值横坐标 (px)", "至下一条纹间距 (px)", "至下一条纹间距 (mm)"
    ])
    for index, peak in enumerate(result['peaks']):
        spacing_px = result['spacings_px'][index] if index < len(result['spacings_px']) else ""
        spacing_mm = result['spacings_mm'][index] if index < len(result['spacings_mm']) else ""
        writer.writerow([index + 1, peak, spacing_px, spacing_mm])
    return output.getvalue().encode("utf-8-sig")


def result_json_bytes(result, physics_result=None):
    """生成包含测量参数和实验计算结果的 JSON。"""
    preprocessing = result["preprocessing"]
    measurement = {
        "图像尺寸 (px)": result["image_size"],
        "标定比例 (px/mm)": result["pixel_per_mm"],
        "亮条纹数量": result["num_fringes"],
        "平均条纹间距 (px)": result["mean_spacing_px"],
        "平均条纹间距 (mm)": result["mean_spacing_mm"],
        "条纹间距标准差 (px)": result["std_spacing_px"],
        "条纹间距标准差 (mm)": result["std_spacing_mm"],
        "峰值横坐标 (px)": result["peaks"],
        "相邻条纹间距 (px)": result["spacings_px"],
        "相邻条纹间距 (mm)": result["spacings_mm"],
        "信号预处理": {
            "背景去除": "已启用" if preprocessing["remove_background"] else "未启用",
            "背景平滑系数": preprocessing["background_sigma"],
            "信号平滑系数": preprocessing["signal_sigma"],
            "信号归一化": "已启用" if preprocessing["normalize_signal"] else "未启用",
        },
    }
    payload = {
        "测量结果": measurement,
        "物理计算结果": _physics_payload_cn(physics_result),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def marked_png_bytes(marked_rgb):
    """将 RGB 标记图编码为 PNG。"""
    output = io.BytesIO()
    Image.fromarray(marked_rgb).save(output, format="PNG")
    return output.getvalue()


def result_pdf_bytes(result, marked_rgb, signal_bundle, physics_result=None):
    """生成包含摘要、标记图和信号曲线的 PDF 报告。"""
    output = io.BytesIO()
    report_font, _ = _chinese_font()
    with PdfPages(output) as pdf:
        summary = plt.figure(figsize=(8.27, 11.69))
        summary.text(
            0.08,
            0.94,
            "双缝干涉条纹间距测量报告",
            fontsize=18,
            weight="bold",
            fontproperties=report_font,
        )
        lines = [
            "图像尺寸：{} × {} px".format(*result['image_size']),
            "检测到的亮条纹数量：{}".format(result['num_fringes']),
            "平均条纹间距：{:.3f} px".format(result['mean_spacing_px']),
            "平均条纹间距：{:.6f} mm".format(result['mean_spacing_mm']),
            "条纹间距标准差：{:.6f} mm".format(result['std_spacing_mm']),
            "标定比例：{:.3f} px/mm".format(result['pixel_per_mm']),
            "背景去除：{}".format(
                "已启用"
                if result['preprocessing']['remove_background']
                else "未启用"
            ),
            "背景平滑系数：{:.2f}".format(
                result['preprocessing']['background_sigma']
            ),
            "信号平滑系数：{:.2f}".format(
                result['preprocessing']['signal_sigma']
            ),
            "信号归一化：{}".format(
                "已启用"
                if result['preprocessing']['normalize_signal']
                else "未启用"
            ),
        ]
        if physics_result:
            lines.extend([
                "",
                "物理计算结果：",
            ])
            physics_payload = _physics_payload_cn(physics_result)
            for label, value in physics_payload.items():
                if isinstance(value, float):
                    lines.append("{}：{:.6g}".format(label, value))
                else:
                    lines.append("{}：{}".format(label, value))
        summary.text(
            0.08,
            0.86,
            "\n".join(lines),
            fontsize=11,
            va="top",
            fontproperties=report_font,
            linespacing=1.5,
        )
        summary.text(
            0.08,
            0.10,
            "说明：测量结果受图像质量和标定精度影响。",
            fontsize=9,
            color="dimgray",
            fontproperties=report_font,
        )
        pdf.savefig(summary, bbox_inches="tight")
        plt.close(summary)

        marked_fig, marked_ax = plt.subplots(figsize=(11.69, 8.27))
        marked_ax.imshow(marked_rgb)
        marked_ax.set_title(
            "检测到的亮条纹位置（红线标记）",
            fontproperties=report_font,
        )
        marked_ax.axis("off")
        marked_fig.tight_layout()
        pdf.savefig(marked_fig, bbox_inches="tight")
        plt.close(marked_fig)

        signal_fig, signal_ax = plt.subplots(figsize=(11.69, 6.5))
        x = signal_bundle['x']
        processed = signal_bundle['processed']
        peaks = signal_bundle['peaks']
        signal_ax.plot(x, processed, color="tab:blue", linewidth=1.2,
                       label="预处理后的信号")
        if peaks is not None and len(peaks):
            signal_ax.plot(
                peaks,
                processed[peaks],
                "rx",
                label="检测到的峰值",
            )
        signal_ax.set_xlabel(
            "水平像素位置",
            fontproperties=report_font,
        )
        signal_ax.set_ylabel(
            "归一化光强",
            fontproperties=report_font,
        )
        signal_ax.grid(alpha=0.25)
        signal_ax.legend(prop=report_font)
        signal_fig.tight_layout()
        pdf.savefig(signal_fig, bbox_inches="tight")
        plt.close(signal_fig)

    return output.getvalue()
