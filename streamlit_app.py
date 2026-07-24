"""双缝干涉条纹间距测量 Streamlit 应用。"""

import base64
import hashlib
import io
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from matplotlib import font_manager
from PIL import Image

from fringe_core import FringeBridge
from physics import (
    compare_spacing,
    screen_distance_m,
    slit_spacing_mm,
    theoretical_spacing_mm,
    wavelength_nm,
)
from report_utils import (
    marked_png_bytes,
    result_csv_bytes,
    result_json_bytes,
    result_pdf_bytes,
)

calibration_click_component = components.declare_component(
    "calibration_click_component",
    path=os.path.join(os.path.dirname(__file__), "calibration_click_component"),
)


def clickable_calibration_image(image, display_width, points, key):
    """显示可点击图像并返回原图像素坐标。"""
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")
    image_data = "data:image/png;base64,{}".format(
        base64.b64encode(image_buffer.getvalue()).decode("ascii")
    )
    return calibration_click_component(
        image_data=image_data,
        display_width=display_width,
        source_width=image.width,
        source_height=image.height,
        points=[list(point) for point in points],
        locked=len(points) >= 2,
        key=key,
        default=None,
    )


def configure_matplotlib_fonts():
    """Select an installed CJK font so chart labels render consistently."""
    preferred_fonts = (
        "Microsoft YaHei",
        "Noto Sans SC",
        "SimHei",
        "DengXian",
    )
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


configure_matplotlib_fonts()


st.set_page_config(
    page_title="双缝干涉测量",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_interface_styles():
    """Apply the approved white-blue optical twin visual system."""
    st.markdown(
        """
        <style>
        :root {
            --lab-ink: #0B2D4A;
            --lab-muted: #55758F;
            --lab-accent: #1769AA;
            --lab-success: #148A82;
            --lab-warning: #D79527;
            --lab-frost: #F7FBFF;
            --lab-glass: rgba(255, 255, 255, 0.64);
            --lab-line: rgba(87, 157, 205, 0.26);
        }

        html, body, [class*="css"] {
            color: var(--lab-ink);
            font-family: "Microsoft YaHei UI", "PingFang SC", sans-serif;
        }

        p, label, small {
            color: #45677F;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 10px;
            font-weight: 650;
            transition:
                color .16s ease,
                background .16s ease,
                border-color .16s ease,
                transform .16s cubic-bezier(.16,1,.3,1);
        }

        .stButton > button:focus-visible,
        .stDownloadButton > button:focus-visible {
            outline: 3px solid rgba(54,184,230,.26);
            outline-offset: 2px;
        }

        .stButton > button:disabled,
        .stButton > button:disabled * {
            color: #7890A1;
            background: rgba(227,237,244,.72);
            border-color: rgba(92,140,172,.18);
            transform: none;
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div {
            border-radius: 10px;
        }

        div[data-baseweb="input"] > div:focus-within,
        div[data-baseweb="select"] > div:focus-within {
            border-color: #1769AA;
            box-shadow: 0 0 0 3px rgba(54,184,230,.15);
        }

        /* Optical Twin Console */
        [data-testid="stSidebar"],
        [data-testid="collapsedControl"],
        [data-testid="stToolbar"],
        footer {
            display: none;
        }

        [data-testid="stFileUploadDropzone"] > div > div > span,
        [data-testid="stFileUploadDropzone"] > div > div > small,
        [data-testid="stFileUploadDropzone"] > button[kind="secondary"] {
            font-size: 0;
        }

        [data-testid="stFileUploadDropzone"] > div > div > span::after {
            content: "将文件拖放到此处";
            display: block;
            font-size: .9rem;
            line-height: 1.4;
        }

        [data-testid="stFileUploadDropzone"] > div > div > small::after {
            content: "单个文件最大 200 MB · 支持 JPG、JPEG、PNG、BMP";
            display: block;
            font-size: .75rem;
            line-height: 1.45;
        }

        [data-testid="stFileUploadDropzone"] > button[kind="secondary"]::after {
            content: "浏览文件";
            font-size: .88rem;
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 50% 18%, rgba(255,255,255,.98) 0 16%, rgba(233,245,253,.88) 52%, rgba(218,235,247,.94) 100%),
                linear-gradient(145deg, #f8fcff, #dcecf7);
        }

        [data-testid="stAppViewContainer"]::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            opacity: .32;
            background:
                linear-gradient(90deg, transparent 0 49.9%, rgba(46,126,184,.08) 50%, transparent 50.1%),
                linear-gradient(0deg, transparent 0 49.9%, rgba(46,126,184,.055) 50%, transparent 50.1%);
            background-size: 72px 72px;
            mask-image: radial-gradient(circle at 50% 35%, black, transparent 76%);
        }

        [data-testid="stHeader"] {
            height: 1.2rem;
            background: transparent;
            border: 0;
        }

        .block-container {
            max-width: 1720px;
            padding: 1.15rem 1.35rem 3.5rem;
        }

        .twin-masthead {
            position: relative;
            display: grid;
            grid-template-columns: minmax(12rem, .8fr) minmax(24rem, 2fr) minmax(12rem, .8fr);
            align-items: center;
            min-height: 4.8rem;
            margin-bottom: .75rem;
            padding: .65rem 1.1rem;
            color: var(--lab-ink);
            background: rgba(255,255,255,.66);
            border: 1px solid rgba(92,160,207,.25);
            border-radius: 18px;
            box-shadow: 0 14px 34px rgba(28,76,111,.09), inset 0 1px 0 rgba(255,255,255,.92);
            backdrop-filter: blur(18px) saturate(128%);
        }

        .twin-masthead::before,
        .twin-masthead::after {
            content: "";
            position: absolute;
            top: 50%;
            width: 11%;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(23,105,170,.5));
        }

        .twin-masthead::before {
            left: 1.2rem;
        }

        .twin-masthead::after {
            right: 1.2rem;
            transform: rotate(180deg);
        }

        .twin-title {
            z-index: 1;
            text-align: center;
        }

        .twin-title h1 {
            margin: 0;
            color: #0b2d4a;
            font-family: Bahnschrift, "Microsoft YaHei UI", sans-serif;
            font-size: clamp(1.35rem, 2.15vw, 2.25rem);
            font-weight: 650;
            letter-spacing: -.025em;
        }

        .twin-title p {
            margin: .2rem 0 0;
            color: #55758f;
            font-size: .76rem;
        }

        .twin-live {
            z-index: 1;
            justify-self: end;
            display: flex;
            align-items: center;
            gap: .55rem;
            color: #2b5877;
            font-size: .75rem;
            font-weight: 650;
        }

        .twin-live-dot {
            width: .58rem;
            height: .58rem;
            border-radius: 50%;
            background: #36b8e6;
            box-shadow: 0 0 0 5px rgba(54,184,230,.12);
            animation: twin-pulse 2.4s cubic-bezier(.16,1,.3,1) infinite;
        }

        @keyframes twin-pulse {
            0%, 100% { box-shadow: 0 0 0 4px rgba(54,184,230,.10); }
            45% { box-shadow: 0 0 0 9px rgba(54,184,230,.02); }
        }

        .lab-stepper {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: .65rem;
            margin: 0 0 .8rem;
            padding: .58rem 1rem;
            background: rgba(255,255,255,.54);
            border: 1px solid rgba(92,160,207,.2);
            border-radius: 14px;
            box-shadow: 0 8px 22px rgba(28,76,111,.06);
            backdrop-filter: blur(16px);
        }

        .lab-step {
            display: flex;
            align-items: baseline;
            gap: .45rem;
            min-width: 0;
            padding: .3rem .15rem;
            color: #6d879c;
            font-size: .73rem;
        }

        .lab-step-label {
            color: #315b76;
            font-weight: 650;
            white-space: nowrap;
        }

        .lab-step-status {
            color: #7890a1;
            font-size: .68rem;
            white-space: nowrap;
        }

        .lab-step.is-current .lab-step-status {
            color: #1769aa;
            font-weight: 700;
        }

        .lab-step.is-complete .lab-step-status {
            color: #148a82;
            font-weight: 700;
        }

        div[data-testid="stHorizontalBlock"] {
            gap: .8rem;
        }

        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            min-width: 0;
            padding: 1rem 1.05rem;
            background: rgba(255,255,255,.64);
            border: 1px solid rgba(87,157,205,.26);
            border-radius: 16px;
            box-shadow: 0 16px 36px rgba(31,82,119,.085), inset 0 1px 0 rgba(255,255,255,.94);
            backdrop-filter: blur(18px) saturate(125%);
        }

        div[data-testid="column"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            padding: 0;
            background: transparent;
            border: 0;
            border-radius: 0;
            box-shadow: none;
            backdrop-filter: none;
        }

        div[data-testid="column"]:has(.twin-viewport-title) {
            position: relative;
            overflow: hidden;
            padding: .85rem;
            background: rgba(246,251,255,.72);
            border-color: rgba(54,184,230,.44);
            border-radius: 22px;
            box-shadow: 0 18px 46px rgba(31,82,119,.12), inset 0 1px 0 rgba(255,255,255,.98);
        }

        div[data-testid="column"]:has(.twin-viewport-title)::before {
            content: "";
            position: absolute;
            inset: 0;
            pointer-events: none;
            opacity: .62;
            background:
                linear-gradient(135deg, rgba(54,184,230,.24) 0 2px, transparent 2px) top left/46px 46px no-repeat,
                linear-gradient(225deg, rgba(54,184,230,.24) 0 2px, transparent 2px) top right/46px 46px no-repeat,
                linear-gradient(45deg, rgba(54,184,230,.24) 0 2px, transparent 2px) bottom left/46px 46px no-repeat,
                linear-gradient(315deg, rgba(54,184,230,.24) 0 2px, transparent 2px) bottom right/46px 46px no-repeat;
        }

        .console-rail-title,
        .matrix-title,
        .twin-viewport-title {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: .7rem;
            margin: 0 0 .8rem;
            padding-bottom: .62rem;
            color: #0d4774;
            border-bottom: 1px solid rgba(76,142,191,.23);
            font-family: Bahnschrift, "Microsoft YaHei UI", sans-serif;
            font-size: .86rem;
            font-weight: 700;
        }

        .console-rail-title span,
        .matrix-title span,
        .twin-viewport-title span {
            color: #6b879c;
            font-size: .66rem;
            font-weight: 650;
            letter-spacing: .06em;
        }

        .twin-stage-meta {
            display: flex;
            flex-wrap: wrap;
            gap: .4rem;
            margin: -.2rem 0 .7rem;
        }

        .twin-chip {
            padding: .26rem .52rem;
            color: #275979;
            background: rgba(225,242,252,.7);
            border: 1px solid rgba(79,151,202,.2);
            border-radius: 999px;
            font-size: .68rem;
            font-weight: 650;
        }

        .twin-chip.is-live {
            color: #075f73;
            background: rgba(208,245,250,.76);
        }

        [data-testid="stImage"] {
            overflow: hidden;
            border-radius: 14px;
            background: #dfeaf2;
            box-shadow: 0 12px 26px rgba(18,52,77,.16);
        }

        [data-testid="stImage"] img {
            display: block;
            width: 100%;
            border-radius: 14px;
            filter: saturate(.92) contrast(1.03);
        }

        .twin-empty-stage {
            display: grid;
            place-items: center;
            min-height: 27rem;
            padding: 2rem;
            text-align: center;
            color: #3f6680;
            background:
                radial-gradient(circle at center, rgba(255,255,255,.9), rgba(220,239,250,.54)),
                linear-gradient(90deg, rgba(74,147,198,.06) 1px, transparent 1px),
                linear-gradient(rgba(74,147,198,.06) 1px, transparent 1px);
            background-size: auto, 34px 34px, 34px 34px;
            border: 1px dashed rgba(67,142,194,.38);
            border-radius: 14px;
        }

        .twin-empty-orbit {
            position: relative;
            width: 11rem;
            height: 4.6rem;
            margin: 0 auto 1.25rem;
        }

        .twin-empty-orbit::before {
            content: "";
            position: absolute;
            left: 1rem;
            right: 1rem;
            top: 50%;
            height: 1px;
            background: linear-gradient(90deg, transparent, #36b8e6, transparent);
            box-shadow: 0 0 12px rgba(54,184,230,.46);
        }

        .twin-empty-orbit i {
            position: absolute;
            top: calc(50% - .45rem);
            width: .9rem;
            height: .9rem;
            border: 2px solid #2b8ec5;
            background: rgba(255,255,255,.9);
            transform: rotate(45deg);
        }

        .twin-empty-orbit i:first-child { left: .7rem; }
        .twin-empty-orbit i:nth-child(2) { left: calc(50% - .45rem); }
        .twin-empty-orbit i:last-child { right: .7rem; }

        .twin-empty-stage h2 {
            margin: 0;
            color: #103d60;
            font-family: Bahnschrift, "Microsoft YaHei UI", sans-serif;
            font-size: 1.25rem;
        }

        .twin-empty-stage p {
            max-width: 46ch;
            margin: .55rem auto 0;
            line-height: 1.65;
        }

        .rail-status {
            display: grid;
            gap: .55rem;
        }

        .rail-status-item {
            padding: .72rem .75rem;
            background: rgba(239,248,254,.64);
            border-top: 1px solid rgba(255,255,255,.9);
            border-radius: 10px;
        }

        .rail-status-item span {
            display: block;
            color: #5d7d93;
            font-size: .68rem;
        }

        .rail-status-item strong {
            display: block;
            margin-top: .22rem;
            color: #0e4774;
            font-family: Bahnschrift, "Microsoft YaHei UI", sans-serif;
            font-size: clamp(.95rem, 1.35vw, 1.25rem);
            font-weight: 700;
            line-height: 1.15;
        }

        .rail-status-item small {
            display: block;
            margin-top: .18rem;
            color: #6b879b;
            font-size: .63rem;
        }

        .rail-bar {
            height: .28rem;
            margin-top: .48rem;
            overflow: hidden;
            background: rgba(74,125,158,.12);
            border-radius: 999px;
        }

        .rail-bar i {
            display: block;
            height: 100%;
            background: linear-gradient(90deg, #1769aa, #36b8e6);
            border-radius: inherit;
        }

        .matrix-divider {
            margin: .85rem 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(67,140,191,.34), transparent);
        }

        .matrix-stat-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0,1fr));
            gap: .5rem;
            margin: .55rem 0 .8rem;
        }

        .matrix-stat {
            padding: .62rem;
            background: rgba(237,247,253,.68);
            border-radius: 10px;
        }

        .matrix-stat span {
            display: block;
            color: #648096;
            font-size: .65rem;
        }

        .matrix-stat strong {
            display: block;
            margin-top: .18rem;
            color: #0e4774;
            font-family: Bahnschrift, "Microsoft YaHei UI", sans-serif;
            font-size: .92rem;
        }

        .stButton > button,
        .stDownloadButton > button {
            min-height: 2.75rem;
            border-color: rgba(65,134,183,.34);
            background: rgba(255,255,255,.72);
            box-shadow: inset 0 1px 0 rgba(255,255,255,.88);
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: #2d8fc6;
            color: #0e5d91;
            background: rgba(241,250,255,.94);
        }

        .stButton > button[kind="primary"] {
            background: #1769aa;
            border-color: #1769aa;
            box-shadow: 0 8px 18px rgba(23,105,170,.18), inset 0 1px 0 rgba(255,255,255,.18);
        }

        .stButton > button[kind="primary"]:not(:disabled),
        .stButton > button[kind="primary"]:not(:disabled) p,
        .stButton > button[kind="primary"]:not(:disabled) span {
            color: #F7FBFF !important;
        }

        .stButton > button[kind="primary"]:not(:disabled):hover {
            color: #F7FBFF !important;
            background: #10558C;
            border-color: #10558C;
        }

        .stButton > button[kind="primary"]:not(:disabled):active {
            color: #F7FBFF !important;
            background: #0B3D68;
            border-color: #0B3D68;
            transform: translateY(1px);
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div,
        [data-testid="stFileUploader"] {
            background: rgba(255,255,255,.7);
            border-color: rgba(69,137,186,.3);
        }

        [data-testid="stFileUploader"] {
            padding: .35rem;
            border-radius: 12px;
        }

        [data-testid="stFileUploadDropzone"] {
            flex-direction: column;
            align-items: stretch;
            gap: .65rem;
            padding: .85rem;
        }

        [data-testid="stFileUploadDropzone"] > div {
            width: 100%;
            align-items: center;
        }

        [data-testid="stFileUploadDropzone"] > button {
            width: 100%;
            margin: 0;
        }

        [data-testid="stFileUploadDropzone"] small {
            line-height: 1.4;
            overflow-wrap: anywhere;
        }

        [data-testid="stAlert"] {
            background: rgba(246,251,255,.78);
            border-color: rgba(67,140,191,.25);
        }

        div[data-testid="stMetric"] {
            min-height: auto;
            padding: .7rem;
            background: rgba(237,247,253,.66);
        }

        .lab-formula {
            width: 100%;
            justify-content: space-between;
            color: #174f77;
            background: rgba(220,241,252,.72);
        }

        @media (max-width: 1100px) {
            .twin-masthead {
                grid-template-columns: 1fr;
                gap: .35rem;
                text-align: center;
            }

            .twin-live {
                justify-self: center;
            }

            .lab-stepper {
                grid-template-columns: repeat(5, minmax(0,1fr));
            }

            .lab-step {
                justify-content: center;
            }

            div[data-testid="stHorizontalBlock"] {
                flex-direction: column !important;
                gap: .7rem;
            }

            div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
                flex: 1 1 100% !important;
                width: 100% !important;
                min-width: 100% !important;
            }
        }

        @media (max-width: 760px) {
            .block-container {
                padding: .75rem .65rem 2.5rem;
            }

            div[data-testid="stHorizontalBlock"] {
                flex-direction: column !important;
                gap: .65rem;
            }

            div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
                flex: 1 1 100% !important;
                width: 100% !important;
                min-width: 100% !important;
                padding: .85rem;
            }

            div[data-testid="column"] div[data-testid="stHorizontalBlock"] {
                flex-direction: column !important;
            }

            .console-rail-title,
            .matrix-title,
            .twin-viewport-title {
                word-break: keep-all;
            }

            .twin-masthead::before,
            .twin-masthead::after {
                display: none;
            }

            .twin-empty-stage {
                min-height: 19rem;
            }

            .lab-stepper {
                grid-template-columns: 1fr;
                gap: .2rem;
            }

            .lab-step {
                justify-content: space-between;
                padding: .4rem .15rem;
            }

        }

        @media (prefers-reduced-motion: reduce) {
            .twin-live-dot {
                animation: none;
            }

            .stButton > button,
            .stDownloadButton > button {
                transition: none;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_workflow_status():
    """显示当前实验流程状态。"""
    steps = [
        ("上传图像", fb.has_image),
        ("尺度标定", st.session_state.calibrated),
        ("条纹分析", st.session_state.last_result is not None),
        ("实验计算", st.session_state.physics_result is not None),
        ("结果导出", st.session_state.export_completed),
    ]
    items = []
    previous_complete = True
    for label, complete in steps:
        if complete:
            status, state_class = "已完成", "is-complete"
        elif previous_complete:
            status, state_class = "正在进行", "is-current"
        else:
            status, state_class = "尚未进行", "is-pending"
        items.append(
            '<div class="lab-step {state}">'
            '<span class="lab-step-label">{label}</span>'
            '<span class="lab-step-status">{status}</span></div>'.format(
                state=state_class, label=label, status=status
            )
        )
        previous_complete = previous_complete and complete
    st.markdown(
        '<div class="lab-stepper">{}</div>'.format("".join(items)),
        unsafe_allow_html=True,
    )


def initialize_state():
    if "fb" not in st.session_state:
        st.session_state.fb = FringeBridge()
    defaults = {
        "calib_points": [],
        "calibrated": False,
        "last_result": None,
        "physics_result": None,
        "export_completed": False,
        "calibration_method": "图上点击标定",
        "calibration_click_version": 0,
        "temp_path": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def clear_dependent_results():
    """清除标定或图像变化后失效的分析、计算和导出状态。"""
    st.session_state.last_result = None
    st.session_state.physics_result = None
    st.session_state.export_completed = False


def rerun_app():
    """兼容新旧 Streamlit 版本的页面重新运行接口。"""
    rerun = getattr(st, "rerun", None)
    if rerun is None:
        rerun = st.experimental_rerun
    rerun()


def load_uploaded_image(uploaded, bridge):
    """Load a newly uploaded image and reset dependent experiment state."""
    if uploaded is None:
        return
    file_bytes = uploaded.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()[:12]
    suffix = os.path.splitext(uploaded.name)[1].lower() or ".png"
    new_path = os.path.join(
        tempfile.gettempdir(), "fringe_{}{}".format(file_hash, suffix)
    )
    if st.session_state.temp_path == new_path:
        return
    with open(new_path, "wb") as file_obj:
        file_obj.write(file_bytes)
    bridge.load_image(new_path)
    st.session_state.temp_path = new_path
    st.session_state.calib_points = []
    st.session_state.calibrated = False
    clear_dependent_results()
    st.session_state.calibration_click_version += 1
    for widget_key in ("manual_p1_x", "manual_p1_y", "manual_p2_x", "manual_p2_y"):
        st.session_state.pop(widget_key, None)


def reset_calibration_selection(bridge):
    """清除标定点以及依赖这些点的实验结果。"""
    bridge.clear_calibration()
    st.session_state.calib_points = []
    st.session_state.calibrated = False
    clear_dependent_results()
    st.session_state.calibration_click_version += 1


def render_console_header():
    """Render the architectural title frame for the optical console."""
    st.markdown(
        """
        <header class="twin-masthead">
            <div aria-hidden="true"></div>
            <div class="twin-title">
                <h1>条纹间距测量平台</h1>
                <p>图像标定 · 条纹诊断 · 物理计算 · 实验留档</p>
            </div>
            <div class="twin-live">
                <i class="twin-live-dot"></i>
                本地实验链路
            </div>
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_input_rail(bridge):
    """Render upload and calibration controls in the left instrument rail."""
    st.markdown(
        '<div class="console-rail-title">图像与标定'
        '<span>图像输入与尺度标定</span></div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "上传干涉条纹图像",
        type=["jpg", "jpeg", "png", "bmp"],
        help="支持 JPG、PNG 与 BMP；图片只在本机处理。",
    )
    if uploaded is None and st.session_state.temp_path is not None:
        bridge.clear_image()
        st.session_state.temp_path = None
        st.session_state.calib_points = []
        st.session_state.calibrated = False
        clear_dependent_results()
        st.session_state.calibration_click_version += 1
    if uploaded is not None:
        try:
            load_uploaded_image(uploaded, bridge)
        except Exception as exc:
            st.error("图片加载失败：{}".format(exc))

    if not bridge.has_image:
        st.info("上传实验图像后，标定控制将在这里启用。")
        st.caption("建议使用条纹清晰、曝光均匀且透视变形较小的原始图片。")
        return

    width, height = bridge.image_size
    st.markdown(
        '<div class="matrix-stat-grid">'
        '<div class="matrix-stat"><span>图像宽度</span><strong>{} px</strong></div>'
        '<div class="matrix-stat"><span>图像高度</span><strong>{} px</strong></div>'
        '</div>'.format(width, height),
        unsafe_allow_html=True,
    )

    calibration_method = st.radio(
        "选择标定方式",
        ["图上点击标定", "手动坐标标定"],
        horizontal=True,
        disabled=st.session_state.calibrated,
        key="calibration_method",
        on_change=reset_calibration_selection,
        args=(bridge,),
    )

    if not st.session_state.calibrated and calibration_method == "图上点击标定":
        st.caption("请在中心条纹图上依次点击两个已知点，系统会自动计算像素距离。")
    elif not st.session_state.calibrated:
        st.caption("请输入两个已知点在原图中的 X、Y 像素坐标。")
        point_1, point_2 = st.columns(2)
        point_1_x = point_1.number_input(
            "点 1 · 横坐标 X",
            min_value=0,
            max_value=max(0, width - 1),
            value=0,
            step=1,
            key="manual_p1_x",
        )
        point_1_y = point_1.number_input(
            "点 1 · 纵坐标 Y",
            min_value=0,
            max_value=max(0, height - 1),
            value=0,
            step=1,
            key="manual_p1_y",
        )
        point_2_x = point_2.number_input(
            "点 2 · 横坐标 X",
            min_value=0,
            max_value=max(0, width - 1),
            value=max(0, width - 1),
            step=1,
            key="manual_p2_x",
        )
        point_2_y = point_2.number_input(
            "点 2 · 纵坐标 Y",
            min_value=0,
            max_value=max(0, height - 1),
            value=0,
            step=1,
            key="manual_p2_y",
        )
        st.session_state.calib_points = [
            (int(point_1_x), int(point_1_y)),
            (int(point_2_x), int(point_2_y)),
        ]

    points = st.session_state.calib_points
    if calibration_method == "图上点击标定" and len(points) == 0:
        st.warning("请在中心条纹图上选择第一个标定点。")
    elif calibration_method == "图上点击标定" and len(points) == 1:
        st.info("点 1：({}, {})".format(*points[0]))
        st.warning("请继续选择第二个标定点。")
    else:
        pixel_distance = float(np.linalg.norm(np.subtract(points[1], points[0])))
        st.markdown(
            '<div class="matrix-stat-grid">'
            '<div class="matrix-stat"><span>点 1</span><strong>({}, {})</strong></div>'
            '<div class="matrix-stat"><span>点 2</span><strong>({}, {})</strong></div>'
            '<div class="matrix-stat"><span>标定线长度</span>'
            '<strong>{:.1f} px</strong></div>'
            '<div class="matrix-stat"><span>当前状态</span><strong>{}</strong></div>'
            '</div>'.format(
                points[0][0],
                points[0][1],
                points[1][0],
                points[1][1],
                pixel_distance,
                "已应用" if st.session_state.calibrated else "待应用",
            ),
            unsafe_allow_html=True,
        )
        if pixel_distance < min(width, height) * 0.1:
            st.warning("标定线较短，选择更远的两个点可以降低点击误差。")

    actual_mm = st.number_input(
        "两点实际距离（mm）",
        min_value=0.001,
        value=10.0,
        step=1.0,
        format="%.3f",
        key="calibration_actual_mm",
    )
    apply_column, clear_column = st.columns(2)
    if apply_column.button(
        "应用标定",
        disabled=len(points) < 2,
        type="primary",
        use_container_width=True,
    ):
        try:
            bridge.set_calibration(points[0], points[1], actual_mm)
            st.session_state.calibrated = True
            clear_dependent_results()
            rerun_app()
        except ValueError as exc:
            st.error(str(exc))
    if clear_column.button("重新选点", use_container_width=True):
        reset_calibration_selection(bridge)
        rerun_app()


def render_twin_viewport(bridge):
    """Render the real fringe image as the center digital twin."""
    state_label = "等待图像"
    if bridge.has_image:
        state_label = "等待标定"
    if st.session_state.calibrated:
        state_label = "标定完成"
    if st.session_state.last_result is not None:
        state_label = "检测结果"

    st.markdown(
        '<div class="twin-viewport-title">实验数字孪生'
        '<span>{}</span></div>'.format(state_label),
        unsafe_allow_html=True,
    )

    if not bridge.has_image:
        st.markdown(
            """
            <div class="twin-empty-stage">
                <div>
                    <div class="twin-empty-orbit"><i></i><i></i><i></i></div>
                    <h2>等待干涉条纹图像</h2>
                    <p>上传图像后，这里会成为实验的中心孪生视口，显示标定点、检测位置与条纹状态。</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        width, height = bridge.image_size
        result = st.session_state.last_result
        stage_mode = "检测标记" if result is not None else "原始条纹"
        calibration_mode = (
            "尺度已锁定" if st.session_state.calibrated else "选择两个标定点"
        )
        st.markdown(
            '<div class="twin-stage-meta">'
            '<span class="twin-chip is-live">{}</span>'
            '<span class="twin-chip">{}</span>'
            '<span class="twin-chip">{} × {} px</span>'
            '</div>'.format(stage_mode, calibration_mode, width, height),
            unsafe_allow_html=True,
        )

        image_rgb = bridge.get_original_image()
        pil_image = Image.fromarray(image_rgb).convert("RGB")
        if (
            not st.session_state.calibrated
            and st.session_state.calibration_method == "图上点击标定"
        ):
            display_width = min(width, 560)
            coordinate = clickable_calibration_image(
                pil_image,
                display_width=display_width,
                points=st.session_state.calib_points,
                key="calibration_click_{}".format(
                    st.session_state.calibration_click_version
                ),
            )
            if coordinate is not None:
                point = (
                    int(coordinate["x"]),
                    int(coordinate["y"]),
                )
                points = st.session_state.calib_points
                selection_changed = False
                if not points:
                    st.session_state.calib_points = [point]
                    selection_changed = True
                elif len(points) == 1 and point != points[0]:
                    st.session_state.calib_points = [points[0], point]
                    selection_changed = True
                if selection_changed:
                    rerun_app()
            if len(st.session_state.calib_points) >= 2:
                st.caption("两个标定点已选定；如需更改，请点击左侧“重新选点”。")
            else:
                st.caption("请直接点击图像选择两个标定点。")
        elif result is not None:
            st.image(
                bridge.get_marked_image(),
                use_column_width=True,
                caption="检测孪生视图 · 红线标记识别到的亮条纹",
            )
        else:
            st.image(
                image_rgb,
                use_column_width=True,
                caption="实验条纹原始视图",
            )

def render_diagnostic_rail(bridge):
    """Render trustworthy experiment diagnostics in the right rail."""
    st.markdown(
        '<div class="console-rail-title">实验诊断'
        '<span>实时实验诊断</span></div>',
        unsafe_allow_html=True,
    )
    if bridge.has_image:
        width, height = bridge.image_size
        image_value = "{} × {} px".format(width, height)
        image_note = "图像已载入"
    else:
        image_value = "—"
        image_note = "等待上传"

    if st.session_state.calibrated:
        calibration_value = "{:.3f}".format(bridge.pixel_per_mm)
        calibration_note = "px / mm"
    else:
        calibration_value = "—"
        calibration_note = "等待尺度标定"

    result = st.session_state.last_result
    if result is None:
        fringe_value = "—"
        spacing_value = "—"
        spread_value = "—"
        spread_width = 0
    else:
        fringe_value = "{} 条".format(result["num_fringes"])
        spacing_value = "{:.4f} mm".format(result["mean_spacing_mm"])
        if result["mean_spacing_mm"] > 0:
            relative_spread = (
                result["std_spacing_mm"] / result["mean_spacing_mm"] * 100.0
            )
        else:
            relative_spread = 0.0
        spread_value = "{:.2f} %".format(relative_spread)
        spread_width = max(4, min(100, int(relative_spread * 8)))

    st.markdown(
        """
        <div class="rail-status">
            <div class="rail-status-item">
                <span>图像分辨率</span><strong>{}</strong><small>{}</small>
            </div>
            <div class="rail-status-item">
                <span>标定比例</span><strong>{}</strong><small>{}</small>
            </div>
            <div class="rail-status-item">
                <span>检测条纹数量</span><strong>{}</strong><small>有效亮峰</small>
            </div>
            <div class="rail-status-item">
                <span>平均条纹间距</span><strong>{}</strong><small>实验测量值</small>
            </div>
            <div class="rail-status-item">
                <span>间距相对离散度</span><strong>{}</strong>
                <div class="rail-bar"><i style="width:{}%"></i></div>
                <small>标准差 ÷ 平均值；越低表示间距越一致</small>
            </div>
        </div>
        """.format(
            image_value,
            image_note,
            calibration_value,
            calibration_note,
            fringe_value,
            spacing_value,
            spread_value,
            spread_width,
        ),
        unsafe_allow_html=True,
    )


def render_analysis_matrix(bridge):
    """Render analysis controls, real results, and the signal trace."""
    st.markdown(
        '<div class="matrix-title">条纹与信号分析'
        '<span>条纹信号分析</span></div>',
        unsafe_allow_html=True,
    )
    if not st.session_state.calibrated:
        st.info("完成尺度标定后，条纹分析参数将在此启用。")
        return

    prominence = st.slider(
        "峰值灵敏度",
        0.005,
        0.5,
        0.05,
        0.005,
        format="%.3f",
        help="越小越容易检测弱峰，也更容易把噪声识别为条纹。",
        key="console_prominence",
    )
    minimum_distance = st.slider(
        "最小峰间距（px）",
        3,
        200,
        10,
        1,
        help="防止同一条亮纹内部被重复识别。",
        key="console_min_distance",
    )
    show_advanced = st.checkbox(
        "显示高级预处理参数",
        value=False,
        key="console_show_advanced",
    )
    if show_advanced:
        clahe_clip = st.slider(
            "局部对比度增强", 0.5, 10.0, 2.0, 0.5, key="console_clahe"
        )
        blur_size = st.slider(
            "二维模糊核", 3, 31, 5, 2, key="console_blur"
        )
        remove_background = st.checkbox(
            "去除缓慢变化的背景", value=True, key="console_remove_background"
        )
        background_sigma = st.slider(
            "背景估计尺度",
            5.0,
            150.0,
            30.0,
            5.0,
            disabled=not remove_background,
            key="console_background_sigma",
        )
        signal_sigma = st.slider(
            "一维平滑强度", 0.0, 10.0, 2.0, 0.5, key="console_signal_sigma"
        )
        normalize_signal = st.checkbox(
            "标准化处理后信号", value=True, key="console_normalize"
        )
    else:
        clahe_clip = 2.0
        blur_size = 5
        remove_background = True
        background_sigma = 30.0
        signal_sigma = 2.0
        normalize_signal = True

    if st.button(
        "执行条纹分析",
        type="primary",
        use_container_width=True,
        key="console_analyze",
    ):
        try:
            st.session_state.last_result = bridge.analyze(
                clahe_clip=clahe_clip,
                blur_size=blur_size,
                prominence_ratio=prominence,
                min_distance=minimum_distance,
                remove_background=remove_background,
                background_sigma=background_sigma,
                signal_sigma=signal_sigma,
                normalize_signal=normalize_signal,
            )
            st.session_state.physics_result = None
            st.session_state.export_completed = False
            rerun_app()
        except RuntimeError as exc:
            st.error(str(exc))

    result = st.session_state.last_result
    if result is None:
        return

    st.markdown(
        '<div class="matrix-stat-grid">'
        '<div class="matrix-stat"><span>亮条纹</span><strong>{} 条</strong></div>'
        '<div class="matrix-stat"><span>平均间距</span><strong>{:.4f} mm</strong></div>'
        '<div class="matrix-stat"><span>像素间距</span><strong>{:.2f} px</strong></div>'
        '<div class="matrix-stat"><span>标准差</span><strong>{:.4f} mm</strong></div>'
        '</div>'.format(
            result["num_fringes"],
            result["mean_spacing_mm"],
            result["mean_spacing_px"],
            result["std_spacing_mm"],
        ),
        unsafe_allow_html=True,
    )

    bundle = bridge.get_signal_bundle()
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 4.2), sharex=True)
    figure.patch.set_alpha(0)
    axes[0].plot(
        bundle["x"],
        bundle["raw"],
        color="#385f7b",
        label="原始投影",
        linewidth=1.1,
    )
    if result["preprocessing"]["remove_background"]:
        axes[0].plot(
            bundle["x"],
            bundle["background"],
            color="#d79527",
            label="估计背景",
            linewidth=1.0,
            alpha=0.85,
        )
    axes[0].set_ylabel("亮度")
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].plot(
        bundle["x"], bundle["processed"], color="#1769aa", linewidth=1.25
    )
    peaks = bundle["peaks"]
    axes[1].plot(
        peaks,
        bundle["processed"][peaks],
        linestyle="none",
        marker="o",
        color="#36b8e6",
        markeredgecolor="#0b5d91",
        markersize=4.5,
        label="检测峰值",
    )
    axes[1].set_xlabel("横向像素位置")
    axes[1].set_ylabel("处理后信号")
    axes[1].legend(fontsize=7, frameon=False)
    for axis in axes:
        axis.set_facecolor((1, 1, 1, 0.18))
        axis.grid(color="#b7d4e6", alpha=0.55, linewidth=0.6)
        axis.tick_params(colors="#536f83", labelsize=7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#9ebed3")
        axis.spines["bottom"].set_color("#9ebed3")
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)


def render_physics_matrix():
    """Render double-slit calculations in the middle information matrix."""
    st.markdown(
        '<div class="matrix-title">双缝物理模型'
        '<span>双缝物理计算</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="lab-formula"><span>小角度近似</span>'
        '<code>Δx = λL / d</code></div>',
        unsafe_allow_html=True,
    )
    measured_default = (
        float(st.session_state.last_result["mean_spacing_mm"])
        if st.session_state.last_result is not None
        else 1.0
    )
    mode_configs = {
        "计算理论条纹间距": {
            "slug": "theory",
            "button": "计算理论值",
            "inputs": (
                ("wave", "波长（nm）", 0.001, 632.8, "%.3f"),
                ("distance", "双缝到屏幕距离（m）", 0.001, 1.0, "%.3f"),
                ("slit", "双缝间距（mm）", 0.001, 0.25, "%.3f"),
            ),
        },
        "反算光波波长": {
            "slug": "wave",
            "button": "反算波长",
            "inputs": (
                ("spacing", "实验条纹间距（mm）", 0.000001, measured_default, "%.6f"),
                ("distance", "双缝到屏幕距离（m）", 0.001, 1.0, "%.3f"),
                ("slit", "双缝间距（mm）", 0.001, 0.25, "%.3f"),
            ),
        },
        "反算双缝间距": {
            "slug": "slit",
            "button": "反算双缝间距",
            "inputs": (
                ("spacing", "实验条纹间距（mm）", 0.000001, measured_default, "%.6f"),
                ("wave", "波长（nm）", 0.001, 632.8, "%.3f"),
                ("distance", "双缝到屏幕距离（m）", 0.001, 1.0, "%.3f"),
            ),
        },
        "反算双缝到屏幕距离": {
            "slug": "screen",
            "button": "反算屏距",
            "inputs": (
                ("spacing", "实验条纹间距（mm）", 0.000001, measured_default, "%.6f"),
                ("wave", "波长（nm）", 0.001, 632.8, "%.3f"),
                ("slit", "双缝间距（mm）", 0.001, 0.25, "%.3f"),
            ),
        },
    }
    mode = st.selectbox(
        "计算目标", list(mode_configs), key="console_physics_mode"
    )
    config = mode_configs[mode]
    values = {
        name: st.number_input(
            label,
            min_value=minimum,
            value=default,
            format=number_format,
            key="console_{}_{}".format(name, config["slug"]),
        )
        for name, label, minimum, default, number_format in config["inputs"]
    }
    try:
        if st.button(
            config["button"],
            type="primary",
            use_container_width=True,
            key="console_calc_{}".format(config["slug"]),
        ):
            if mode == "计算理论条纹间距":
                theoretical = theoretical_spacing_mm(
                    values["wave"], values["distance"], values["slit"]
                )
                payload = {
                    "mode": mode,
                    "wavelength_nm": values["wave"],
                    "screen_distance_m": values["distance"],
                    "slit_spacing_mm": values["slit"],
                    "theoretical_spacing_mm": theoretical,
                }
                if st.session_state.last_result is not None:
                    comparison = compare_spacing(measured_default, theoretical)
                    payload["measured_spacing_mm"] = measured_default
                    payload.update(comparison)
            elif mode == "反算光波波长":
                payload = {
                    "mode": mode,
                    "spacing_mm": values["spacing"],
                    "screen_distance_m": values["distance"],
                    "slit_spacing_mm": values["slit"],
                    "wavelength_nm": wavelength_nm(
                        values["spacing"], values["distance"], values["slit"]
                    ),
                }
            elif mode == "反算双缝间距":
                payload = {
                    "mode": mode,
                    "spacing_mm": values["spacing"],
                    "wavelength_nm": values["wave"],
                    "screen_distance_m": values["distance"],
                    "slit_spacing_mm": slit_spacing_mm(
                        values["spacing"], values["wave"], values["distance"]
                    ),
                }
            else:
                payload = {
                    "mode": mode,
                    "spacing_mm": values["spacing"],
                    "wavelength_nm": values["wave"],
                    "slit_spacing_mm": values["slit"],
                    "screen_distance_m": screen_distance_m(
                        values["spacing"], values["wave"], values["slit"]
                    ),
                }
            st.session_state.physics_result = payload
            st.session_state.export_completed = False
            rerun_app()
    except ValueError as exc:
        st.error(str(exc))

    physics_result = st.session_state.physics_result
    if not physics_result:
        return
    labels = {
        "wavelength_nm": "波长（nm）",
        "screen_distance_m": "屏距（m）",
        "slit_spacing_mm": "双缝间距（mm）",
        "spacing_mm": "实验间距（mm）",
        "theoretical_spacing_mm": "理论间距（mm）",
        "measured_spacing_mm": "测量间距（mm）",
        "absolute_error_mm": "绝对差（mm）",
        "relative_error_percent": "相对误差（%）",
    }
    items = []
    for key, value in physics_result.items():
        if key == "mode":
            continue
        value_text = "{:.6g}".format(value) if isinstance(value, float) else str(value)
        items.append(
            '<div class="matrix-stat"><span>{}</span><strong>{}</strong></div>'.format(
                labels.get(key, key), value_text
            )
        )
    st.markdown(
        '<div class="matrix-stat-grid">{}</div>'.format("".join(items)),
        unsafe_allow_html=True,
    )


def mark_export_completed():
    """记录用户已经下载至少一种实验结果。"""
    st.session_state.export_completed = True


def render_export_matrix(bridge):
    """Render the result package in the final information matrix."""
    st.markdown(
        '<div class="matrix-title">结果与实验留档'
        '<span>测量结果导出</span></div>',
        unsafe_allow_html=True,
    )
    result = st.session_state.last_result
    if result is None:
        st.info("条纹分析完成后，可在这里生成完整实验记录。")
        st.markdown(
            """
            <div class="rail-status">
                <div class="rail-status-item">
                    <span>CSV / JSON</span><strong>测量数据</strong><small>用于复核与二次处理</small>
                </div>
                <div class="rail-status-item">
                    <span>PNG</span><strong>检测标记图</strong><small>保留亮条纹识别位置</small>
                </div>
                <div class="rail-status-item">
                    <span>PDF</span><strong>实验报告</strong><small>摘要、图像与信号曲线</small>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    bundle = bridge.get_signal_bundle()
    marked = bridge.get_marked_image()
    physics_result = st.session_state.physics_result
    st.download_button(
        "下载测量数据（CSV）",
        result_csv_bytes(result),
        file_name="条纹间距测量数据.csv",
        mime="text/csv",
        on_click=mark_export_completed,
        use_container_width=True,
    )
    st.download_button(
        "下载完整数据（JSON）",
        result_json_bytes(result, physics_result),
        file_name="条纹间距完整数据.json",
        mime="application/json",
        on_click=mark_export_completed,
        use_container_width=True,
    )
    st.download_button(
        "下载检测标记图（PNG）",
        marked_png_bytes(marked),
        file_name="亮条纹检测标记图.png",
        mime="image/png",
        on_click=mark_export_completed,
        use_container_width=True,
    )
    st.download_button(
        "下载实验报告（PDF）",
        result_pdf_bytes(result, marked, bundle, physics_result),
        file_name="双缝干涉实验报告.pdf",
        mime="application/pdf",
        on_click=mark_export_completed,
        use_container_width=True,
    )
    st.caption("所有文件均由本地实验数据即时生成。")


def render_optical_twin_console(bridge):
    """Compose the approved B topology with a real fringe-image twin."""
    inject_interface_styles()
    render_console_header()
    workflow_slot = st.empty()

    input_rail, twin_view, diagnostic_rail = st.columns([1.12, 3.2, 1.12])
    with input_rail:
        render_input_rail(bridge)
    with twin_view:
        render_twin_viewport(bridge)
    with diagnostic_rail:
        render_diagnostic_rail(bridge)

    with workflow_slot.container():
        render_workflow_status()

    st.markdown('<div class="matrix-divider"></div>', unsafe_allow_html=True)
    analysis_matrix, physics_matrix, export_matrix = st.columns([1.35, 1, 1])
    with analysis_matrix:
        render_analysis_matrix(bridge)
    with physics_matrix:
        render_physics_matrix()
    with export_matrix:
        render_export_matrix(bridge)


initialize_state()
fb = st.session_state.fb
render_optical_twin_console(fb)
