"""双缝干涉条纹间距测量核心算法（无 GUI 依赖）。"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


class FringeBridge:
    """条纹分析引擎：加载图片、标定、分析，纯算法无 GUI"""

    def __init__(self):
        self._image_rgb = None
        self._gray = None
        self._blur_cache = None
        self._raw_signal_cache = None
        self._background_cache = None
        self._hist_cache = None
        self._peaks_cache = None
        self._pixel_per_mm = None
        self._calib_points = []
        self._image_w = 0
        self._image_h = 0

    # ---- 图片加载 ----

    def load_image(self, path):
        """加载图片，支持中文路径"""
        bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("图片读取失败，请检查文件路径或格式")
        self._image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        self._image_h, self._image_w = bgr.shape[:2]
        self._blur_cache = None
        self._raw_signal_cache = None
        self._background_cache = None
        self._hist_cache = None
        self._peaks_cache = None
        self._pixel_per_mm = None
        self._calib_points = []
        return True

    def clear_image(self):
        """清除当前图片以及所有依赖该图片的状态。"""
        self.__init__()

    # ---- 属性 ----

    @property
    def has_image(self):
        return self._image_rgb is not None

    @property
    def is_calibrated(self):
        return self._pixel_per_mm is not None

    @property
    def image_size(self):
        return (self._image_w, self._image_h)

    @property
    def pixel_per_mm(self):
        return self._pixel_per_mm

    # ---- 标定 ----

    def set_calibration(self, p1, p2, actual_mm):
        """p1, p2: 图像上两个点的像素坐标; actual_mm: 实际物理距离(mm)"""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        pix_dist = np.sqrt(dx ** 2 + dy ** 2)
        if pix_dist == 0:
            raise ValueError("两个标定点不能重合")
        if actual_mm <= 0:
            raise ValueError("实际长度必须大于 0")
        self._pixel_per_mm = pix_dist / actual_mm
        self._calib_points = [list(p1), list(p2)]
        self._peaks_cache = None
        self._hist_cache = None
        self._raw_signal_cache = None
        self._background_cache = None

    def clear_calibration(self):
        """清除标定和已有分析结果。"""
        self._pixel_per_mm = None
        self._calib_points = []
        self._peaks_cache = None
        self._hist_cache = None
        self._raw_signal_cache = None
        self._background_cache = None

    # ---- 分析 ----

    def analyze(self, clahe_clip=2.0, clahe_grid=8, blur_size=5,
                prominence_ratio=0.05, min_distance=10,
                remove_background=True, background_sigma=30.0,
                signal_sigma=2.0, normalize_signal=True):
        """执行条纹分析，返回测量结果字典"""
        if not self.has_image:
            raise RuntimeError("请先加载图片")
        if not self.is_calibrated:
            raise RuntimeError("请先完成标定")

        # CLAHE 增强
        clahe = cv2.createCLAHE(clipLimit=clahe_clip,
                                tileGridSize=(clahe_grid, clahe_grid))
        enhanced = clahe.apply(self._gray)

        # 高斯模糊 (核大小强制奇数)
        k = blur_size if blur_size % 2 == 1 else blur_size + 1
        self._blur_cache = cv2.GaussianBlur(enhanced, (k, k), 0)

        # 垂直压缩为一维投影信号。后续寻峰只使用处理后信号，
        # 同时保留原始投影和背景，方便界面展示处理效果。
        raw_signal = cv2.reduce(
            self._blur_cache, 0, cv2.REDUCE_AVG
        ).reshape(-1).astype(np.float64)
        self._raw_signal_cache = raw_signal.copy()

        signal = raw_signal.copy()
        self._background_cache = np.zeros_like(signal)

        if remove_background:
            sigma = max(float(background_sigma), 1.0)
            self._background_cache = gaussian_filter1d(signal, sigma=sigma)
            signal = signal - self._background_cache

        if signal_sigma > 0:
            signal = gaussian_filter1d(signal, sigma=float(signal_sigma))

        if normalize_signal:
            signal = signal - np.mean(signal)
            std = float(np.std(signal))
            if std > 1e-12:
                signal = signal / std

        self._hist_cache = signal

        # find_peaks 寻峰
        signal_range = float(np.ptp(self._hist_cache))
        if signal_range <= 1e-12:
            raise RuntimeError("一维亮度信号几乎没有变化，无法识别条纹")
        prominence = signal_range * prominence_ratio
        self._peaks_cache, _ = find_peaks(
            self._hist_cache, prominence=prominence, distance=min_distance
        )
        self._peaks_cache.sort()

        if len(self._peaks_cache) < 2:
            self._peaks_cache = None
            raise RuntimeError("检测到的条纹不足，请调整分析参数")

        # 计算间距
        spacings_pix = np.diff(self._peaks_cache)
        mean_pix = float(np.mean(spacings_pix))
        mean_mm = mean_pix / self._pixel_per_mm

        return {
            'num_fringes': len(self._peaks_cache),
            'mean_spacing_px': mean_pix,
            'mean_spacing_mm': mean_mm,
            'std_spacing_px': float(np.std(spacings_pix)),
            'std_spacing_mm': float(np.std(spacings_pix) / self._pixel_per_mm),
            'pixel_per_mm': self._pixel_per_mm,
            'peaks': self._peaks_cache.tolist(),
            'spacings_px': spacings_pix.tolist(),
            'spacings_mm': (spacings_pix / self._pixel_per_mm).tolist(),
            'calib_points': self._calib_points,
            'image_size': self.image_size,
            'preprocessing': {
                'remove_background': bool(remove_background),
                'background_sigma': float(background_sigma),
                'signal_sigma': float(signal_sigma),
                'normalize_signal': bool(normalize_signal),
            },
        }

    # ---- 图像获取 ----

    def get_marked_image(self):
        """返回标记了条纹线的 RGB 图像"""
        if not self.has_image:
            return None
        img = self._image_rgb.copy()
        if self._peaks_cache is not None:
            h = img.shape[0]
            for x in self._peaks_cache:
                cv2.line(img, (int(round(x)), 0),
                         (int(round(x)), h), (255, 0, 0), 2)
        return img

    def get_original_image(self):
        return self._image_rgb if self.has_image else None

    def get_signal_data(self):
        """返回 (x, 处理后信号, peaks)，兼容原有界面。"""
        if self._hist_cache is None:
            return None, None, None
        x = np.arange(len(self._hist_cache))
        peaks = self._peaks_cache.copy() if self._peaks_cache is not None else None
        return x, self._hist_cache, peaks

    def get_signal_bundle(self):
        """返回原始投影、估计背景、处理后信号和峰值。"""
        if self._hist_cache is None:
            return None
        return {
            'x': np.arange(len(self._hist_cache)),
            'raw': self._raw_signal_cache.copy(),
            'background': self._background_cache.copy(),
            'processed': self._hist_cache.copy(),
            'peaks': (
                self._peaks_cache.copy()
                if self._peaks_cache is not None else None
            ),
        }
