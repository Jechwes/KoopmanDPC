"""Overlay two flights filmed from the same static camera, one drone blue, the other orange.

    python overlay_videos.py SPC.mp4 KSPC.mp4 -o SPC_vs_KSPC.mp4
    python overlay_videos.py SPC.mp4 KSPC.mp4 --preview previews/   # stills + both paths

The first clip is drawn blue and the second orange (the colours of plot_compare.py).
Both are resampled to the same frame rate, so the clips must start at the same moment.

How it works:
  1. background = per-pixel median of the clip; the moving drone drops out of it. Pixels
     that change in many frames (a live screen, flicker) are ignored from then on
  2. the clips may be exported at different sizes or crops, so the first view is
     registered onto the second (SIFT matches + RANSAC homography)
  3. per frame, the drone is the blob that differs from the background, tracked from
     frame to frame so that noise and other motion are not picked up
  4. the drone pixels are tinted and pasted onto the shared background

Needs: pip install opencv-python-headless imageio-ffmpeg   (or ffmpeg on PATH)
"""

import argparse
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

COLORS = ((0xd6, 0x78, 0x2a),       # BGR of #2a78d6, SPC blue
          (0x34, 0x68, 0xeb))       # BGR of #eb6834, KSPC orange


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        exe = shutil.which("ffmpeg")
        if exe is None:
            sys.exit("needs ffmpeg: pip install imageio-ffmpeg, or put ffmpeg on PATH")
        return exe


class Clip:
    """Frames of a video at a fixed rate, decoded by ffmpeg (which also applies rotation)."""

    def __init__(self, ff, path, fps):
        self.ff, self.path, self.fps = ff, path, fps
        png = subprocess.run([ff, "-loglevel", "error", "-i", path, "-frames:v", "1",
                              "-f", "image2pipe", "-vcodec", "png", "-"],
                             capture_output=True, check=True).stdout
        self.h, self.w = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR).shape[:2]

    def frames(self, step=1):
        p = subprocess.Popen([self.ff, "-loglevel", "error", "-i", self.path,
                              "-vf", f"fps={self.fps}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                             stdout=subprocess.PIPE)
        n, i = self.w * self.h * 3, 0
        try:
            while True:
                buf = p.stdout.read(n)
                if len(buf) < n:
                    return
                if i % step == 0:
                    yield np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)
                i += 1
        finally:
            if p.poll() is None:            # stopped early (the other clip is shorter)
                p.kill()
            p.stdout.close()
            p.wait()


def background(clip, thresh, every_s=0.5, busy_frac=0.25):
    """Median background, the number of frames, and the pixels that change too often to
    be the drone (it passes any one spot in only a few frames)."""
    step = max(1, round(every_s * clip.fps))
    sample = np.stack(list(clip.frames(step)))
    bg = np.median(sample, axis=0).astype(np.uint8)
    changed = np.zeros(bg.shape[:2], np.float32)
    for f in sample:
        changed += cv2.absdiff(f, bg).max(axis=2) > thresh
    busy = cv2.dilate((changed / len(sample) > busy_frac).astype(np.uint8), np.ones((15, 15), np.uint8))
    return bg, busy.astype(bool), len(sample) * step


def register(src, dst, ignore_src, ignore_dst):
    """Homography taking the src view onto the dst view."""
    sift = cv2.SIFT_create(4000)
    k1, d1 = sift.detectAndCompute(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), (~ignore_src).astype(np.uint8) * 255)
    k2, d2 = sift.detectAndCompute(cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY), (~ignore_dst).astype(np.uint8) * 255)
    good = [m for m, n in cv2.BFMatcher().knnMatch(d1, d2, k=2) if m.distance < 0.75 * n.distance]
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    H, inl = cv2.findHomography(p1, p2, cv2.RANSAC, 2.0)
    inl = inl.ravel().astype(bool)
    err = np.linalg.norm(cv2.perspectiveTransform(p1[inl][None], H)[0] - p2[inl], axis=1)
    print(f"registration: {inl.sum()} matching features, median error {np.median(err):.2f} px")
    if inl.sum() < 30:
        print("  WARNING: few matches, check that both clips show the same view")
    return H


class Tracker:
    """The drone is the blob that differs from the background near where it was last.
    The biggest change in the frame (re)acquires it, e.g. when it sits on the ground at
    the start."""

    def __init__(self, bg, busy, thresh, min_area=8, max_lost=15):
        self.bg, self.busy, self.thresh = bg, busy, thresh
        w = bg.shape[1]
        self.radius, self.merge_r = 0.06 * w, 0.03 * w      # search and part-merging radius
        self.min_area, self.max_lost = min_area, max_lost
        self.pos, self.vel, self.lost = None, np.zeros(2), 0

    def __call__(self, frame):
        d = cv2.GaussianBlur(cv2.absdiff(frame, self.bg).max(axis=2), (3, 3), 0)
        m = ((d > self.thresh) & ~self.busy).astype(np.uint8)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
        area = stats[:, cv2.CC_STAT_AREA]
        cand = [j for j in range(1, n) if area[j] >= self.min_area]
        if self.pos is not None and self.lost <= self.max_lost:
            pred = self.pos + self.vel
            dist = {j: np.hypot(*(cent[j] - pred)) for j in cand}
            cand = [j for j in cand if dist[j] < self.radius]
            score = {j: area[j] / (1 + dist[j] / self.radius) for j in cand}
        else:
            score = {j: area[j] for j in cand}
        if not cand:
            self.lost += 1
            return None, None
        c = cent[max(cand, key=score.get)]
        parts = [j for j in range(1, n) if area[j] >= 2 and np.hypot(*(cent[j] - c)) < self.merge_r]
        mask = np.isin(lab, parts)
        ys, xs = np.nonzero(mask)
        new = np.array([xs.mean(), ys.mean()])
        if self.pos is not None and self.lost == 0:
            self.vel = 0.5 * self.vel + 0.5 * (new - self.pos)
        else:
            self.vel = np.zeros(2)
        self.pos, self.lost = new, 0
        return mask, new


def tint(frame, alpha, color):
    """The drone in a flat colour, keeping some of its own shading."""
    shade = 0.55 + 0.45 * cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    return shade[..., None] * np.float32(color), alpha[..., None]


def soft(mask, grow=2):
    m = cv2.dilate(mask.astype(np.uint8) * 255, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))
    return cv2.GaussianBlur(m, (5, 5), 0).astype(np.float32) / 255


def legend(w, h, labels):
    """RGBA legend in the bottom-left corner, drawn with PIL for a decent font."""
    from PIL import Image, ImageDraw, ImageFont
    size = max(14, round(h / 28))
    font = None
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf",
                 "/Library/Fonts/Arial.ttf"):
        if os.path.exists(path):
            font = ImageFont.truetype(path, size)
            break
    font = font or ImageFont.load_default()
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    pad, r = size // 2, size // 2.6
    rows = [(lab, (c[2], c[1], c[0])) for lab, c in zip(labels, COLORS)]
    tw = max(dr.textlength(lab, font=font) for lab, _ in rows)
    bw, bh = int(pad * 3 + 2 * r + tw), int(pad * (len(rows) + 1) + size * len(rows))
    x0, y0 = pad, h - pad - bh
    dr.rounded_rectangle([x0, y0, x0 + bw, y0 + bh], radius=pad, fill=(255, 255, 255, 200))
    for k, (lab, rgb) in enumerate(rows):
        cy = y0 + pad + k * (size + pad) + size / 2
        dr.ellipse([x0 + pad, cy - r, x0 + pad + 2 * r, cy + r], fill=rgb + (255,))
        dr.text((x0 + 2 * pad + 2 * r, cy), lab, font=font, fill=(30, 30, 30, 255), anchor="lm")
    a = np.asarray(img).astype(np.float32) / 255
    return a[..., 2::-1] * 255, a[..., 3:]                         # BGR, alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("first", help="drawn blue (e.g. SPC.mp4)")
    ap.add_argument("second", help="drawn orange (e.g. KSPC.mp4)")
    ap.add_argument("-o", "--out", default=None, help="output video, default next to the first clip")
    ap.add_argument("--labels", nargs=2, default=None, help="legend text, default the file names")
    ap.add_argument("--no-legend", action="store_true")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--thresh", type=int, default=30, help="colour difference that counts as the drone")
    ap.add_argument("--preview", default=None, help="write stills and the tracked paths here instead")
    args = ap.parse_args()

    ff = ffmpeg_exe()
    clips = [Clip(ff, p, args.fps) for p in (args.first, args.second)]
    labels = args.labels or [os.path.splitext(os.path.basename(p))[0] for p in (args.first, args.second)]
    for c, lab in zip(clips, labels):
        print(f"{lab}: {c.w}x{c.h}")

    # the larger frame is the canvas; the other clip is warped onto it
    ref = int(clips[1].w * clips[1].h >= clips[0].w * clips[0].h)
    oth = 1 - ref
    W, Hh = clips[ref].w, clips[ref].h
    bgs, busy, nfr = zip(*(background(c, args.thresh) for c in clips))
    print(f"backgrounds from {nfr[0]} and {nfr[1]} frames; ignoring "
          f"{100 * busy[0].mean():.1f} / {100 * busy[1].mean():.1f} % of the pixels as always changing")
    H = register(bgs[oth], bgs[ref], busy[oth], busy[ref])
    warps = [np.eye(3), np.eye(3)]
    warps[oth] = H

    trackers = [Tracker(b, m, args.thresh) for b, m in zip(bgs, busy)]
    lg = None if args.no_legend else legend(W, Hh, labels)
    base = bgs[ref].astype(np.float32)

    if args.preview:
        os.makedirs(args.preview, exist_ok=True)
        stills = {int(t * args.fps) for t in (3, 10, 15, 20, 25, 30, 40, 50)}
        writer = None
    else:
        out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.first)),
                                       f"{labels[0]}_vs_{labels[1]}_overlay.mp4")
        writer = subprocess.Popen([ff, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                   "-s", f"{W}x{Hh}", "-r", f"{args.fps}", "-i", "-",
                                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                                   "-preset", "medium", "-movflags", "+faststart", out],
                                  stdin=subprocess.PIPE)

    tracks, found = ([], []), [0, 0]
    for k, pair in enumerate(zip(*(c.frames() for c in clips))):
        img = base.copy()
        for i in (0, 1):                     # the second clip is drawn on top
            mask, pos = trackers[i](pair[i])
            if mask is None:
                continue
            found[i] += 1
            frame, alpha = pair[i], soft(mask)
            if i == oth:
                frame = cv2.warpPerspective(frame, H, (W, Hh), flags=cv2.INTER_LINEAR)
                alpha = cv2.warpPerspective(alpha, H, (W, Hh), flags=cv2.INTER_LINEAR)
                pos = cv2.perspectiveTransform(pos.reshape(1, 1, 2).astype(np.float32), H)[0, 0]
            tracks[i].append((k, *pos))
            layer, a = tint(frame, alpha, COLORS[i])
            img = img * (1 - a) + layer * a
        if lg is not None:
            img = img * (1 - lg[1]) + lg[0] * lg[1]
        img = np.clip(img, 0, 255).astype(np.uint8)
        if writer is not None:
            writer.stdin.write(img.tobytes())
        elif k in stills:
            cv2.imwrite(os.path.join(args.preview, f"overlay_t{k / args.fps:04.1f}s.png"), img)

    n = k + 1
    print(f"{n} frames ({n / args.fps:.1f} s); drone found in {100 * found[0] / n:.0f} % ({labels[0]}) "
          f"and {100 * found[1] / n:.0f} % ({labels[1]}) of them")
    if writer is not None:
        writer.stdin.close()
        writer.wait()
        print(f"saved {out}")
    else:
        # both tracked paths on the background: jumps here mean the tracker lost the drone
        img = bgs[ref].copy()
        for i in (0, 1):
            pts = np.array([(x, y) for _, x, y in tracks[i]], np.int32)
            cv2.polylines(img, [pts], False, COLORS[i], 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(args.preview, "paths.png"), img)
        print(f"saved stills and paths.png in {args.preview}")


if __name__ == "__main__":
    main()
