# -*- coding: utf-8 -*-
"""Длинный файл (README, раздел 5). Без модели: движок подменён, своя временная папка.

    python3 "_общие модули/stt/tests/test_longfile.py"
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stt import longfile as lf  # noqa: E402

SR = lf.SR


class FakeDecoder:
    """Кусок -> один сегмент «t<начало куска в секундах>». Кусок, начинающийся в long_at, - «упирается в предел
    токенов» один раз; в repeat_at - зацикливается; в silent_at - тишина."""

    def __init__(self, audio, long_at=(), repeat_at=(), silent_at=()):
        self.audio, self.calls, self.seen = audio, 0, []
        self.long_at, self.repeat_at, self.silent_at = set(long_at), set(repeat_at), set(silent_at)
        self.cut_once = set()

    def where(self, clip):
        return int(round(float(clip[0]))) if len(clip) else 0   # в тестовом звуке значение = секунда записи

    def __call__(self, clips):
        self.calls += 1
        out = []
        for c in clips:
            t = self.where(c)
            self.seen.append(t)
            dur = len(c) / SR
            if t in self.repeat_at:
                out.append({"repeat": True})
            elif t in self.silent_at:
                out.append({"silent": True})
            elif t in self.long_at and t not in self.cut_once:
                self.cut_once.add(t)
                out.append({"segments": [{"start": 0.0, "end": dur / 2, "text": f" t{t}a"}], "rest": dur / 2})
            else:
                out.append({"segments": [{"start": 0.0, "end": dur, "text": f" t{t}"}], "rest": None})
        return out


def ramp(sec):
    """Звук, где каждый отсчёт = номер секунды: FakeDecoder узнаёт, откуда кусок."""
    return np.repeat(np.arange(sec, dtype=np.float32), SR)


SPANS = [(0, 12), (13, 25), (26, 29), (31, 55), (56, 58), (60, 90), (92, 100)]


class Chunks(unittest.TestCase):
    def test_merge_in_pauses_up_to_30s(self):
        ch = lf.merge_chunks(SPANS)
        self.assertEqual(ch, [(0, 29), (31, 58), (60, 90), (92, 100)])
        for a, b in ch:
            self.assertLessEqual(b - a, 30)
            self.assertTrue(any(a == x for x, _ in SPANS) and any(b == y for _, y in SPANS))  # границы - в паузах


class Transcribe(unittest.TestCase):
    def test_order_offsets_and_rest(self):
        audio = ramp(100)
        dec = FakeDecoder(audio, long_at={31})
        segs = lf.transcribe(audio, SPANS, dec, batch=2)
        self.assertEqual([s["text"].strip() for s in segs], ["t0", "t31a", "t44", "t60", "t92"])
        self.assertEqual(segs[1]["start"], 31.0)
        self.assertEqual(segs[2]["start"], 44.5)                 # хвост (с середины куска 31-58) - со своего места
        self.assertTrue(all(segs[k]["start"] <= segs[k + 1]["start"] for k in range(len(segs) - 1)))

    def test_repeat_goes_to_fallback_and_silence_is_skipped(self):
        audio = ramp(100)
        dec = FakeDecoder(audio, repeat_at={60}, silent_at={92})
        got = []
        segs = lf.transcribe(audio, SPANS, dec, fallback=lambda c: got.append(len(c)) or
                             [{"start": 1.0, "end": 2.0, "text": " запасной"}])
        self.assertEqual(got, [30 * SR])
        self.assertEqual([s["text"].strip() for s in segs], ["t0", "t31", "запасной"])
        self.assertEqual(segs[2]["start"], 61.0)

    def test_progress_reaches_one(self):
        seen = []
        lf.transcribe(ramp(100), SPANS, FakeDecoder(None), batch=3, prog=seen.append)
        self.assertEqual(seen[0], 0.0)
        self.assertEqual(seen[-1], 1.0)


class Resume(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.key = {"file": "/x/запись.wav", "size": 1, "mtime": 2, "model": "turbo", "lang": "ru"}

    def tearDown(self):
        self.tmp.cleanup()

    def test_cancel_then_resume_gives_same_text(self):
        audio = ramp(100)
        whole = lf.transcribe(audio, SPANS, FakeDecoder(audio, long_at={31}), batch=1)
        dec1, n = FakeDecoder(audio, long_at={31}), {"k": 0}

        def stop():
            n["k"] += 1
            return n["k"] > 2                                    # две пачки - и сбой / «Остановить»
        with self.assertRaises(lf.Cancelled):
            lf.transcribe(audio, SPANS, dec1, batch=1, store=lf.Store(self.root, self.key), cancel=stop)
        share = lf.saved_share(self.root, self.key)
        self.assertGreater(share, 0)
        self.assertLess(share, 1)
        dec2 = FakeDecoder(audio, long_at={31})
        again = lf.transcribe(audio, SPANS, dec2, batch=1, store=lf.Store(self.root, self.key))
        self.assertEqual(again, whole)
        self.assertNotIn(0, dec2.seen)                           # готовое заново не распознаётся

    def test_other_settings_start_over_and_broken_line_is_ignored(self):
        audio = ramp(100)
        st = lf.Store(self.root, self.key)
        lf.transcribe(audio, SPANS, FakeDecoder(audio), store=st)
        with open(st.dir / "done.jsonl", "a", encoding="utf-8") as f:
            f.write('{"i": 3, "segm')                            # строка, оборванная сбоем
        self.assertEqual(len(lf.Store(self.root, self.key).open(lf.merge_chunks(SPANS))), 4)
        other = dict(self.key, lang="en")
        self.assertIsNone(lf.saved_share(self.root, other))
        dec = FakeDecoder(audio)
        lf.transcribe(audio, SPANS, dec, store=lf.Store(self.root, other))
        self.assertEqual(len(dec.seen), 4)
        st.clear()
        self.assertFalse(st.dir.exists())

    def test_prune_removes_only_old(self):
        import os
        audio = ramp(100)
        old, new = lf.Store(self.root, self.key), lf.Store(self.root, dict(self.key, lang="en"))
        for st in (old, new):
            lf.transcribe(audio, SPANS, FakeDecoder(audio), store=st)
        for f in [old.dir, *old.dir.iterdir()]:
            os.utime(f, (1, 1))
        lf.prune(self.root, days=30)
        self.assertFalse(old.dir.exists())
        self.assertTrue(new.dir.exists())

    def test_file_key_changes_with_file(self):
        p = self.root / "a.wav"
        p.write_bytes(b"1")
        k1 = lf.file_key(p, model="turbo")
        p.write_bytes(b"12")
        self.assertNotEqual(k1, lf.file_key(p, model="turbo"))
        self.assertEqual(json.loads(json.dumps(k1)), k1)


class Tokens(unittest.TestCase):
    TB, EOT = 1000, 999

    def text(self, toks):
        return " " + " ".join(f"w{t}" for t in toks)

    def test_finished_answer(self):
        tb = self.TB
        segs, rest = lf._timestamp_segments([tb, 1, 2, tb + 50, tb + 50, 3, tb + 100], tb, self.EOT, self.text)
        self.assertIsNone(rest)
        self.assertEqual([(s["start"], s["end"]) for s in segs], [(0.0, 1.0), (1.0, 2.0)])

    def test_unfinished_answer_gives_rest(self):
        tb = self.TB
        segs, rest = lf._timestamp_segments([tb, 1, 2, tb + 50, tb + 50, 3, 4], tb, self.EOT, self.text)
        self.assertEqual(len(segs), 1)
        self.assertEqual(rest, 1.0)                              # дописать с 1-й секунды куска


if __name__ == "__main__":
    unittest.main()
