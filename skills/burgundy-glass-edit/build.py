import sys, os, json, subprocess, functools
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

D = '/tmp/claude-0/vid'
W, H, FPS = 1080, 1920, 30
SRC = f'{D}/in.mov'

# ---------------- scenes (start, end, intro, big word, icon, pop time) ----------------
SCENES = [
 (0.50, 9.45,  "سر راح يغيّر حياتك للأبد", "السر", "lock", 3.38),
 (9.50, 13.55, "لو أحد قالها لي قبل ٥ سنوات", "180°", "autorenew", 12.40),
 (13.60, 21.15, "دخلي عالي.. بس", "فلوسي تضيع", "money_off", 14.10),
 (21.20, 27.08, "تعلّمت الاستثمار وكل شي", "تدخل وتطلع", "sync_alt", 26.25),
 (27.12, 35.36, "الأفكار والصورة الذاتية", "المعتقدات", "psychology", 28.55),
 (35.40, 44.98, "قبل كل صرف.. وش بيحسسني؟", "المشاعر", "favorite", 36.55),
 (45.02, 56.60, "طاقة الاستقبال", "الأنوثة", "spa", 48.28),
 (56.64, 64.86, "مضايقة ومهمومة", "القرآن", "menu_book", 60.58),
 (64.90, 69.70, "﴿فَابْتَغُوا عِندَ اللَّهِ الرِّزْقَ﴾", "عند الله", "auto_awesome", 65.78),
 (69.74, 74.74, "وهنا لقيت", "المفتاح", "key", 71.10),
 (74.78, 78.85, "بحثنا في القرآن كامل", "COUNT6236", "search", 76.50),
 (78.88, 81.92, "الرزق انذكر في", "109 آية", "search", 81.00),
 (81.96, 85.02, "ولا آية تربط الرزق", "بالسعي", "block", 84.54),
 (85.06, 88.80, "ولعت في راسي لمبة", "الرزّاق", "lightbulb", 88.15),
 (88.84, 91.30, "طيب ليش نتخذ الأسباب؟", "الأسباب", "help", 90.18),
 (91.36, 99.52, "كل العلوم لها دور", "الربط", "hub", 96.00),
 (99.56, 106.18, "استراتيجية هرمية", "GrowRich", "change_history", 101.35),
 (106.22, 111.52, "من الداخل إلى الخارج", "من الداخل", "self_improvement", 106.50),
 (111.56, 118.36, "محتاسة وما تدرين من وين تبدين؟", "رتّبي دخلك", "account_balance_wallet", 112.50),
 (118.40, 121.77, "اكتبي في التعليقات", "GrowRich", "chat", 118.92),
]

FIX = {"فانتي": "فأنتي", "امنيت": "تمنّيت", "قل": "قال", "الانثة": "الأنوثة", "ركتر": "بروكتر",
       "الميتاهيلت": "الميتاهيلث", "جروريتش": "GrowRich", "فذا": "فإذا", "فاكتب": "فاكتبي",
       "ايضا": "أيضاً", "اكيد": "أكيد", "الافكار": "الأفكار", "الاسباب": "الأسباب", "الى": "إلى",
       "انت": "إنتي", "اش": "إيش", "جدا": "جداً", "وافكارك": "وأفكارك", "الاصطناعي": "الاصطناعي"}

# ---------------- fonts ----------------
def font(name, size, axes=None):
    f = ImageFont.truetype(f'{D}/fonts/{name}.ttf', size, layout_engine=ImageFont.Layout.RAQM)
    if axes is not None:
        f.set_variation_by_axes(axes)
    return f

def is_ar(t): return any('؀' <= c <= 'ۿ' or 'ﭐ' <= c <= 'ﻼ' for c in t)

def text_mask(text, f, pad=40):
    d = 'rtl' if is_ar(text) else 'ltr'
    tmp = ImageDraw.Draw(Image.new('L', (10, 10)))
    x0, y0, x1, y1 = tmp.textbbox((0, 0), text, font=f, direction=d)
    im = Image.new('L', (x1 - x0 + 2 * pad, y1 - y0 + 2 * pad), 0)
    ImageDraw.Draw(im).text((pad - x0, pad - y0), text, font=f, fill=255, direction=d)
    return np.asarray(im).astype(np.float32) / 255

def hexc(h): h = h.lstrip('#'); return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)], np.float32) / 255

COPPER = [hexc('#FFF3E6'), hexc('#F2C6A6'), hexc('#E6A98A'), hexc('#B8734A'), hexc('#8C4A2C')]

def vgrad(h, stops):
    t = np.linspace(0, 1, h)[:, None]
    n = len(stops) - 1
    idx = np.minimum((t * n).astype(int), n - 1)
    fr = t * n - idx
    a = np.array(stops)[idx[:, 0]]; b = np.array(stops)[idx[:, 0] + 1]
    return a * (1 - fr) + b * fr  # (h,3)

def rgba_layer(rgb, a):  # rgb HxWx3 (0-1 RGB), a HxW
    return np.dstack([rgb, a]).astype(np.float32)

def copper_word(text, max_w=760, size=210):
    f = None
    while True:
        f = font('Alexandria', size, [900])
        m = text_mask(text, f)
        if m.shape[1] - 80 <= max_w or size < 90: break
        size -= 8
    h, w = m.shape
    ys, = np.where(m.max(1) > 0.05); top, bot = ys[0], ys[-1]
    g = np.zeros((h, w, 3), np.float32)
    g[top:bot + 1] = vgrad(bot - top + 1, COPPER)[:, None, :]
    g[:top] = COPPER[0]; g[bot:] = COPPER[-1]
    # bevel highlight on upper edges
    sh = np.zeros_like(m); sh[3:] = m[:-3]
    hl = np.clip(m - sh, 0, 1)[..., None]
    g = g * (1 - hl * 0.8) + hexc('#FFFFFF') * hl * 0.8
    # dark lower edge
    sh2 = np.zeros_like(m); sh2[:-3] = m[3:]
    dk = np.clip(m - sh2, 0, 1)[..., None]
    g = g * (1 - dk * 0.5) + hexc('#4A1E10') * dk * 0.5
    # drop shadow
    shadow = cv2.GaussianBlur(np.roll(m, 12, 0), (0, 0), 10) * 0.65
    rgb = g * m[..., None] + hexc('#14040A') * (shadow * (1 - m))[..., None]
    a = np.clip(m + shadow * (1 - m), 0, 1)
    rgb = np.where(a[..., None] > 0, rgb / np.maximum(a[..., None], 1e-4), 0)
    return rgba_layer(rgb, a)

def plain_text(text, size, color='#FFF4E8', fname='Cairo', wght=500, shadow=0.55):
    f = font(fname, size, [wght, 0]) if fname == 'Cairo' else font(fname, size)
    m = text_mask(text, f)
    sh = cv2.GaussianBlur(np.roll(m, 4, 0), (0, 0), 6) * shadow
    a = np.clip(m + sh * (1 - m), 0, 1)
    rgb = (hexc(color) * m[..., None] + hexc('#0A0205') * (sh * (1 - m))[..., None]) / np.maximum(a[..., None], 1e-4)
    return rgba_layer(rgb, a)

def icon_layer(name, size=230):
    f = ImageFont.truetype(f'{D}/fonts/Icons.ttf', size, layout_engine=ImageFont.Layout.RAQM)
    axes = {a['name']: a for a in f.get_variation_axes()}
    vals = []
    for a in f.get_variation_axes():
        n = a['name'].lower() if isinstance(a['name'], str) else a['name'].decode().lower()
        vals.append({'fill': 0, 'grade': 0, 'optical size': 48, 'weight': 250}.get(n, a['default']))
    f.set_variation_by_axes(vals)
    m = text_mask(name, f, pad=50)
    glow = cv2.GaussianBlur(m, (0, 0), 14) * 0.9
    h = m.shape[0]
    g = vgrad(h, COPPER[1:])[:, None, :] * np.ones((1, m.shape[1], 1), np.float32)
    a = np.clip(m + glow * (1 - m), 0, 1)
    rgb = (g * m[..., None] + hexc('#C0503A') * (glow * (1 - m))[..., None]) / np.maximum(a[..., None], 1e-4)
    return rgba_layer(rgb, a)

# ---------------- layout ----------------
PX0, PY0, PX1, PY1, PR = 150, 190, 930, 1660, 66

def rrect_mask(x0, y0, x1, y1, r, ss=4):
    m = np.zeros((H * 1, W * 1), np.uint8)
    im = Image.new('L', (W * ss, H * ss), 0)
    ImageDraw.Draw(im).rounded_rectangle([x0 * ss, y0 * ss, x1 * ss, y1 * ss], r * ss, fill=255)
    return np.asarray(im.resize((W, H), Image.LANCZOS)).astype(np.float32) / 255

def build_static():
    S = {}
    pa = rrect_mask(PX0, PY0, PX1, PY1, PR)
    inner = rrect_mask(PX0 + 3, PY0 + 3, PX1 - 3, PY1 - 3, PR - 3)
    ring = np.clip(pa - inner, 0, 1)
    S['pa'] = pa
    rg = np.zeros((H, W, 3), np.float32)
    rg[PY0:PY1 + 1] = vgrad(PY1 - PY0 + 1, [hexc('#FFF3E6'), hexc('#E6A98A'), hexc('#B8734A'), hexc('#E6A98A'), hexc('#8C4A2C')])[:, None, :]
    S['ring'] = rgba_layer(rg, ring)
    # inner glow (top-left stronger)
    ig = np.clip(pa - cv2.GaussianBlur(inner, (0, 0), 18), 0, 1) * pa
    yy, xx = np.mgrid[0:H, 0:W]
    bias = np.clip(1.2 - ((xx - PX0) / (PX1 - PX0) * 0.6 + (yy - PY0) / (PY1 - PY0) * 0.8), 0.2, 1)
    S['iglow'] = ig * bias * 0.55
    # outer burgundy glow
    S['oglow'] = cv2.GaussianBlur(pa, (0, 0), 60) * 0.55 * (1 - pa)
    # tint gradient for glass
    tg = np.zeros((H, W, 3), np.float32)
    tg[:] = vgrad(H, [hexc('#8A2234'), hexc('#7A1A2A'), hexc('#5A0F1F'), hexc('#4A0C19')])[:, None, :]
    S['tint'] = tg
    # vignette
    v = ((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2
    S['vig'] = np.clip(1 - v * 0.55, 0.35, 1)[..., None].astype(np.float32)
    # notch
    S['notch'] = rrect_mask(470, PY0 + 22, 610, PY0 + 34, 6) * 0.35
    return S

def scene_assets(i):
    s, e, intro, word, icon, pop = SCENES[i]
    A = {'intro': plain_text(intro, 46, wght=500), 'icon': icon_layer(icon)}
    if not word.startswith('COUNT'):
        A['word'] = copper_word(word)
    return A

@functools.lru_cache(maxsize=64)
def count_word(n): return copper_word(f'{n} آية')

# ---------------- captions ----------------
def build_chunks():
    d = json.load(open(f'{D}/words.json'))
    chunks = []
    for seg in d:
        ws = []
        for k, w in enumerate(seg['words']):
            t = w['w'].strip()
            if t == 'من' and k + 1 < len(seg['words']) and seg['words'][k + 1]['w'].strip() == 'ضايقة':
                continue
            if t == 'ضايقة': t = 'مضايقة'
            t = FIX.get(t, t)
            ws.append((t, w['s'], w['e']))
        for j in range(0, len(ws), 4):
            part = ws[j:j + 4]
            if len(part) == 1 and chunks and j > 0:
                chunks[-1]['w'].extend(part); chunks[-1]['e'] = part[-1][2]; continue
            chunks.append({'w': part, 's': part[0][1], 'e': part[-1][2]})
    for k in range(len(chunks) - 1):
        chunks[k]['e'] = min(max(chunks[k]['e'], chunks[k]['s'] + 0.3), chunks[k + 1]['s'])
    return chunks

CAP_F = None
@functools.lru_cache(maxsize=256)
def caption_img(key, active):
    words = key.split('\u0001')
    f = font('Cairo', 54, [700, 0])
    parts = []
    for k, t in enumerate(words):
        col = '#F2B48E' if k == active else '#FFFFFF'
        parts.append((t, col))
    # render RTL: draw words right-to-left
    tmp = ImageDraw.Draw(Image.new('L', (10, 10)))
    sizes = []
    for t, c in parts:
        d = 'rtl' if is_ar(t) else 'ltr'
        b = tmp.textbbox((0, 0), t, font=f, direction=d); sizes.append((b, d))
    gap = 16
    tw = sum(b[2] - b[0] for b, _ in sizes) + gap * (len(parts) - 1)
    hh = 110
    im = Image.new('RGBA', (tw + 80, hh + 40), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    x = tw + 40
    for (t, c), (b, d) in zip(parts, sizes):
        w = b[2] - b[0]
        x -= w
        dr.text((x - b[0], 20 + 10), t, font=f, fill=c, direction=d)
        x -= gap
    arr = np.asarray(im).astype(np.float32) / 255
    a = arr[..., 3]
    sh = cv2.GaussianBlur(np.roll(a, 4, 0), (0, 0), 7) * 0.75
    A = np.clip(a + sh * (1 - a), 0, 1)
    rgb = (arr[..., :3] * a[..., None] + hexc('#0A0205') * (sh * (1 - a))[..., None]) / np.maximum(A[..., None], 1e-4)
    return rgba_layer(rgb, A)

# ---------------- helpers ----------------
def over(canvas, layer, cx, cy, alpha=1.0, scale=1.0, blur=0.0):
    """canvas RGB float HxWx3 ; layer RGBA ; place centered at (cx,cy)"""
    if alpha <= 0.003: return
    L = layer
    if abs(scale - 1) > 0.002:
        L = cv2.resize(L, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    if blur > 0.3:
        L = cv2.GaussianBlur(L, (0, 0), blur)
    h, w = L.shape[:2]
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if X1 <= X0 or Y1 <= Y0: return
    sub = L[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    a = sub[..., 3:4] * alpha
    canvas[Y0:Y1, X0:X1] = canvas[Y0:Y1, X0:X1] * (1 - a) + sub[..., :3] * a

def ease_pop(t):  # t in frames since pop
    if t < 0: return 0.0, 0.0
    T1, T2 = 6, 11
    if t < T1: s = 1.08 * (t / T1)
    elif t < T2: s = 1.08 - 0.08 * ((t - T1) / (T2 - T1))
    else: s = 1.0
    return max(s, 0.05), min(1.0, t / 5)

def scene_at(t):
    for i, sc in enumerate(SCENES):
        if sc[0] <= t < sc[1] + 0.001: return i
    return None

# ---------------- per-frame compose ----------------
class Comp:
    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks.python import vision, BaseOptions
        self.mp = mp
        self.seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=f'{D}/models/seg.tflite'),
            running_mode=vision.RunningMode.VIDEO, output_confidence_masks=True))
        self.S = build_static()
        self.assets = {}
        self.chunks = build_chunks()
        self.prev = None
        self.ts = 0

    def mask(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.ts += 34
        r = self.seg.segment_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), self.ts)
        p = 1 - r.confidence_masks[0].numpy_view().astype(np.float32)
        if self.prev is not None: p = 0.55 * p + 0.45 * self.prev
        self.prev = p
        m = cv2.resize(p, (W, H), interpolation=cv2.INTER_CUBIC)
        m = np.clip((m - 0.38) / 0.24, 0, 1); m = m * m * (3 - 2 * m)
        return cv2.GaussianBlur(m, (0, 0), 1.6)

    def asset(self, i):
        if i not in self.assets:
            self.assets = {k: v for k, v in self.assets.items() if abs(k - i) <= 1}
            self.assets[i] = scene_assets(i)
        return self.assets[i]

    def frame(self, fbgr, n):
        t = n / FPS
        S = self.S
        m = self.mask(fbgr)
        f = cv2.cvtColor(fbgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        # --- background grade (burgundy, dark)
        lum = (f @ np.array([0.3, 0.59, 0.11], np.float32))[..., None]
        bg = (lum * 0.42 + f * 0.12) * np.array([1.0, 0.42, 0.5], np.float32) + hexc('#1E050B') * 0.9
        bg = bg + hexc('#9E2135') * S['oglow'][..., None]
        # --- frosted glass from frame without the person
        sm = cv2.resize(f, (270, 480), interpolation=cv2.INTER_AREA)
        mb = 1 - cv2.resize(m, (270, 480), interpolation=cv2.INTER_AREA)
        num = cv2.GaussianBlur(sm * mb[..., None], (0, 0), 9); den = cv2.GaussianBlur(mb, (0, 0), 9)[..., None]
        fill = num / np.maximum(den, 1e-3)
        fill = cv2.resize(cv2.GaussianBlur(fill, (0, 0), 3), (W, H), interpolation=cv2.INTER_CUBIC)
        flum = (fill @ np.array([0.3, 0.59, 0.11], np.float32))[..., None]
        glass = S['tint'] * 0.62 + (flum * 0.6 + fill * 0.4) * 0.30 * np.array([1, 0.6, 0.65], np.float32) + 0.06
        # person shadow onto glass
        shs = cv2.GaussianBlur(cv2.resize(m, (270, 480), interpolation=cv2.INTER_AREA), (0, 0), 4.5)
        shs = cv2.resize(shs, (W, H)); M = np.float32([[1, 0, 10], [0, 1, 26]])
        shs = cv2.warpAffine(shs, M, (W, H)) * 0.45
        glass = glass * (1 - shs[..., None])
        pa = S['pa'][..., None]
        c = bg * (1 - pa) + glass * pa
        c = c + hexc('#E6A98A') * (S['iglow'] * 0.35)[..., None]
        c = c * (1 - S['notch'][..., None]) + 0.0
        ring = S['ring']; c = c * (1 - ring[..., 3:4]) + ring[..., :3] * ring[..., 3:4]
        # --- scene elements behind the person
        i = scene_at(t)
        if i is not None:
            s, e, intro, word, icon, pop = SCENES[i]
            A = self.asset(i)
            fin = (t - s) * FPS; fout = (e - t) * FPS
            vis = min(1.0, fin / 7, max(0.0, fout / 6))
            bl = max(0.0, 12 * (1 - fin / 7)) / 2
            over(c, A['intro'], 540, 300, alpha=vis, blur=bl)
            ip = (t - s) * FPS
            sc_i = 0.92 + 0.08 * min(1, ip / 10)
            over(c, A['icon'], 790, 830, alpha=vis * min(1, ip / 8) * 0.95, scale=sc_i)
            pf = (t - pop) * FPS + 2
            sc, al = ease_pop(pf)
            if al > 0:
                if word.startswith('COUNT'):
                    target = int(word[5:]); k = min(1.0, max(0.0, pf / 50))
                    k = 1 - (1 - k) ** 3
                    L = count_word(int(round(target * k / 7)) * 7 if k < 1 else target)
                else:
                    L = A['word']
                over(c, L, 540, 545, alpha=al * vis, scale=sc, blur=max(0, 6 * (1 - pf / 5)) if pf < 5 else 0)
        # --- person on top
        pm = m[..., None]
        person = f * np.array([1.02, 0.99, 0.97], np.float32)
        c = c * (1 - pm) + person * pm
        # --- captions (front)
        for ch in self.chunks:
            if ch['s'] - 0.05 <= t < ch['e'] + 0.15:
                act = 0
                for k, (_, ws, we) in enumerate(ch['w']):
                    if t >= ws - 0.03: act = k
                key = '\u0001'.join(w for w, _, _ in ch['w'])
                img = caption_img(key, act)
                cf = (t - ch['s'] + 0.05) * FPS
                over(c, img, 540, 1440, alpha=min(1, cf / 4), scale=0.96 + 0.04 * min(1, cf / 5))
                break
        # --- finishing
        c = c * S['vig']
        rng = np.random.default_rng(n)
        c = c + (rng.standard_normal((H // 2, W // 2, 1)).astype(np.float32).repeat(2, 0).repeat(2, 1)) * 0.012
        return (np.clip(c, 0, 1) * 255).astype(np.uint8)


def run_range(a, b, out):
    comp = Comp()
    cap = cv2.VideoCapture(SRC)
    warm = max(0, a - 6)
    cap.set(cv2.CAP_PROP_POS_FRAMES, warm)
    for n in range(warm, a):
        ok, fr = cap.read(); comp.mask(fr)
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
                          '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '17', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
    for n in range(a, b):
        ok, fr = cap.read()
        if not ok: break
        p.stdin.write(comp.frame(fr, n).tobytes())
        if (n - a) % 150 == 0: print(f'{out}: {n - a}/{b - a}', flush=True)
    p.stdin.close(); p.wait()


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'stills':
        comp = Comp(); cap = cv2.VideoCapture(SRC)
        ts = [float(x) for x in sys.argv[2].split(',')]
        outs = []
        for t in ts:
            n = int(t * FPS); comp.prev = None
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n - 4))
            for k in range(max(0, n - 4), n):
                ok, fr = cap.read(); comp.mask(fr)
            ok, fr = cap.read()
            outs.append(cv2.resize(cv2.cvtColor(comp.frame(fr, n), cv2.COLOR_RGB2BGR), (540, 960)))
        cv2.imwrite(sys.argv[3], np.hstack(outs))
    elif mode == 'range':
        run_range(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
