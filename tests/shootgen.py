"""Write small synthetic shoots: real JPEGs with EXIF that bracketeering groups for real.

No real images or GPS: rendered scenes (gradients + shapes), with sharpness and exposure controlled per
frame, so routing and decisions can be tested end to end.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

EXIF_IFD = 0x8769


@dataclass
class Frame:
    name: str
    second: float  # capture time offset from 12:00:00
    scene: int = 0  # which rendered subject
    blur: float = 0.0  # Gaussian blur radius: higher = softer
    gain: float = 1.0  # brightness multiplier
    exposure: tuple[int, int] = (1, 250)
    bias: float = 0.0  # exposure compensation, EV
    mode: int = 0  # EXIF ExposureMode: 0 auto, 1 manual, 2 auto bracket
    blown_sky: float = 0.0  # paint this fraction of the frame's height, from the top, pure white


def render(scene: int, size=(480, 320)):
    from PIL import Image, ImageDraw

    w, h = size
    im = Image.new("RGB", size)
    d = ImageDraw.Draw(im)
    for x in range(w):
        v = int(40 + 140 * x / w) if scene % 2 == 0 else int(180 - 140 * x / w)
        d.line([(x, 0), (x, h)], fill=(v, v, min(255, v + 30)))
    if scene % 3 == 0:
        d.rectangle([40, 160, 180, 300], fill=(90, 60, 30))
        d.ellipse([300, 40, 380, 120], fill=(230, 220, 160))
        for i in range(0, w, 12):
            d.line([(i, 300), (i + 6, 320)], fill=(20, 20, 20), width=2)
    elif scene % 3 == 1:
        d.polygon([(20, 300), (240, 60), (460, 300)], fill=(40, 90, 50))
        d.rectangle([340, 30, 450, 100], fill=(200, 60, 60))
        for i in range(0, h, 10):
            d.line([(0, i), (60, i + 5)], fill=(240, 240, 240), width=1)
    else:
        for i in range(8):
            d.ellipse([30 + 55 * i, 100 + (i % 3) * 40, 80 + 55 * i, 150 + (i % 3) * 40], fill=(30 * i, 200 - 20 * i, 120))
        d.rectangle([0, 260, w, h], fill=(70, 70, 90))
    return im


def jpeg_bytes(frame: Frame) -> bytes:
    import io

    from PIL import Image, ImageFilter
    from PIL.TiffImagePlugin import IFDRational

    im = render(frame.scene)
    if frame.blur:
        im = im.filter(ImageFilter.GaussianBlur(frame.blur))
    if frame.gain != 1.0:
        im = im.point(lambda p: max(0, min(255, int(p * frame.gain))))
    if frame.blown_sky:
        from PIL import ImageDraw

        ImageDraw.Draw(im).rectangle([0, 0, im.width, int(im.height * frame.blown_sky)], fill=(255, 255, 255))
    whole = int(frame.second)
    sub = int(round((frame.second - whole) * 100))
    exif = Image.Exif()
    exif[0x010F] = "Canon"
    exif[0x0110] = "Canon EOS R5"
    exif[0x0112] = 1
    e = exif.get_ifd(EXIF_IFD)
    e[0x829A] = IFDRational(*frame.exposure)  # ExposureTime
    e[0x829D] = IFDRational(8, 1)  # FNumber
    e[0x8822] = 3
    e[0x8827] = 100  # ISO
    e[0x9003] = f"2024:05:01 12:{whole // 60:02d}:{whole % 60:02d}"
    e[0x9011] = "+02:00"
    e[0x9291] = f"{sub:02d}"
    b = Fraction(frame.bias).limit_denominator(3)
    e[0x9204] = IFDRational(b.numerator, b.denominator)
    e[0x920A] = IFDRational(24, 1)
    e[0xA402] = frame.mode
    e[0xA403] = 0
    e[0xA431] = "012345678901"
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=92, exif=exif.tobytes())
    return buf.getvalue()


def write_shoot(root: Path, frames: list[Frame]) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    out = {}
    for f in frames:
        p = root / f"{f.name}.JPG"
        p.write_bytes(jpeg_bytes(f))
        out[f.name] = p
    return out


def burst(prefix: str, start: int, t0: float, n: int, scene: int, blurs: list[float] | None = None, pace: float = 0.1) -> list[Frame]:
    blurs = blurs or [0.0] * n
    return [Frame(f"{prefix}{start + i:04d}", t0 + i * pace, scene, blur=blurs[i]) for i in range(n)]


def bracket(prefix: str, start: int, t0: float, scene: int, evs=(0.0, -1.0, 1.0), pace: float = 0.3) -> list[Frame]:
    out = []
    for i, ev in enumerate(evs):
        den = int(round(250 / (2 ** ev)))
        out.append(Frame(f"{prefix}{start + i:04d}", t0 + i * pace, scene, gain=2 ** (ev * 0.6), exposure=(1, den), bias=ev, mode=2))
    return out
