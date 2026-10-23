# -*- coding: utf-8 -*-
"""Очередь видеокарты `.heavy-lock` - общий модуль программ Андрея (описание - README.md рядом).

Замок - папка `.heavy-lock` (mkdir атомарен) рядом с папками проектов; в ней `owner` (прежняя строка
«<Программа> <дата> pid N» - её читают старые копии) и `owner.json` (кто, что делает, с какого
времени, человек или фон). Очередь - файлы в `.heavy-lock-queue/`, там же журнал.

  * владелец - живой процесс: чужой замок снимается, только если его процесс мёртв (не по возрасту);
  * очередь честная: сначала люди, потом фон, внутри - по времени постановки;
  * фон отдаёт замок ждущему человеку на своей контрольной точке (`checkpoint`);
  * сессия человека без действий 15 минут отпускает замок сама (`idle_check`), при продолжении - `resume`;
  * «Запустить сейчас» - работа поверх чужого замка, видна в очереди как `now-…`.

Командная строка (её вызывает heavy_lock.sh): take / release / status / yield / run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

try:
    VERSION = Path(__file__).with_name("VERSION").read_text(encoding="utf-8").strip()
except OSError:
    VERSION = "1.0.0"

HUMAN, BACKGROUND = "human", "background"
IDLE_AFTER = 15 * 60        # простой сессии человека, после которого замок отпускается (стандарт 2.5)
WRITING_GRACE = 10          # папка без owner моложе этого - владелец как раз пишет файл
LEGACY_STALE = 3 * 3600     # owner без номера процесса старше этого - брошен (прежнее правило)
JOURNAL_MAX = 1_000_000     # журнал больше этого - оставляем вторую половину
MARKER = "Очередь проектов.md"   # по нему находится папка «Разработка ПО»
_DAYS = r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun)"


# ============================================================ пути
def default_path(path_env: str | None = None) -> Path:
    """HEAVY_LOCK → прежняя переменная программы → «Разработка ПО»/.heavy-lock → собранная программа
    (установщик Windows, .app) - общая временная папка системы → <проект>/../.heavy-lock."""
    env = os.environ.get("HEAVY_LOCK") or (os.environ.get(path_env) if path_env else None)
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / MARKER).exists():
            return p / ".heavy-lock"
    if getattr(sys, "frozen", False):                 # не мусорить в папке установки; одна очередь на машину
        import tempfile
        base = Path(tempfile.gettempdir()) / "heavy-lock"
        try:
            base.mkdir(exist_ok=True)
        except OSError:
            pass
        return base / ".heavy-lock"
    return here.parents[2].parent / ".heavy-lock"     # <проект>/common/lock/heavy_lock.py


def queue_dir(lock: Path) -> Path:
    return lock.with_name(lock.name + "-queue")


# ============================================================ процессы
def alive(pid: int | None) -> bool:
    """Жив ли процесс. На Windows os.kill(pid, 0) - Ctrl+C всей группе, поэтому через WinAPI."""
    if not pid or pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return code.value == 259                 # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


# ============================================================ кто держит
@dataclass
class Holder:
    """Владелец замка или ожидающий в очереди."""
    program: str
    pid: int | None = None
    what: str = ""
    who: str = HUMAN
    since: dt.datetime | None = None
    host: str = ""
    token: str = ""
    legacy: bool = False        # прочитан из старой строки owner (без owner.json)
    kind: str = "hold"          # hold - держит; wait - в очереди; now - работает поверх
    ts: float = 0.0             # время постановки в очередь (для порядка)
    file: Path | None = field(default=None, repr=False)

    @property
    def since_text(self) -> str:
        return since_text(self.since)

    @property
    def background(self) -> bool:
        return self.who == BACKGROUND

    def to_json(self) -> dict:
        return {"program": self.program, "pid": self.pid, "what": self.what, "who": self.who,
                "since": self.since.isoformat(timespec="seconds") if self.since else None,
                "since_text": self.since_text, "host": self.host, "kind": self.kind}


def since_text(since: dt.datetime | None, now: dt.datetime | None = None) -> str:
    """«16:31» - сегодня, «22.10 16:31» - раньше."""
    if since is None:
        return "?"
    now = now or dt.datetime.now()
    return f"{since:%H:%M}" if since.date() == now.date() else f"{since:%d.%m %H:%M}"


def _parse_legacy(txt: str) -> Holder:
    """Строка прежних копий: «Понятный документ Fri Oct 23 19:05:25 MSK 2026 pid 56980»."""
    m = re.match(r"(.+?)\s+" + _DAYS + r"\b(.*?)(?:pid (\d+))?\s*$", txt.splitlines()[0] if txt else "")
    pid = re.search(r"pid (\d+)", txt)
    program = (m.group(1) if m else txt.split(" pid ")[0]).strip() or "?"
    since = None
    if m:
        stamp = " ".join(w for w in (m.group(2) + m.group(3)).split() if not re.fullmatch(r"[A-Z]{2,5}|[+-]\d{4}", w))
        try:
            since = dt.datetime.strptime(stamp.strip(), "%a %b %d %H:%M:%S %Y")
        except ValueError:
            since = None
    return Holder(program=program, pid=int(pid.group(1)) if pid else None, since=since, legacy=True)


def _read_holder(lock: Path) -> Holder | None:
    """Владелец по owner.json (новые копии) или по строке owner (старые). None - файла нет."""
    try:
        d = json.loads((lock / "owner.json").read_text(encoding="utf-8"))
        since = dt.datetime.fromisoformat(d["since"]) if d.get("since") else None
        return Holder(program=d.get("program") or "?", pid=d.get("pid"), what=d.get("what", ""),
                      who=d.get("who", HUMAN), since=since, host=d.get("host", ""), token=d.get("token", ""))
    except (OSError, ValueError, KeyError, TypeError):
        pass
    try:
        txt = (lock / "owner").read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return None
    h = _parse_legacy(txt)
    if h.since is None:
        h.since = _mtime(lock)
    return h


def _mtime(p: Path) -> dt.datetime | None:
    try:
        return dt.datetime.fromtimestamp(p.stat().st_mtime)
    except OSError:
        return None


def _age(p: Path) -> float | None:
    try:
        return time.time() - p.stat().st_mtime
    except OSError:
        return None


def _stale(lock: Path, h: Holder | None) -> bool:
    """Брошен ли замок: процесс владельца мёртв; без номера процесса - старше 3 ч (README, 4.1)."""
    if h is not None and h.pid:
        return not alive(h.pid)
    age = _age(lock)
    return age is not None and age > LEGACY_STALE


def status(path: Path | None = None) -> Holder | None:
    """Кто держит замок (живой владелец). None - свободен (или брошен - его снимет первый, кто возьмёт)."""
    lock = Path(path) if path else default_path()
    if not lock.is_dir():
        return None
    h = _read_holder(lock)
    if h is None:
        age = _age(lock)
        if age is not None and age < WRITING_GRACE:
            return Holder(program="?", since=_mtime(lock))       # пишется прямо сейчас
        if _stale(lock, None):
            return None
        return Holder(program="?", since=_mtime(lock))
    return None if _stale(lock, h) else h


# ============================================================ очередь
def _entries(lock: Path) -> list[Holder]:
    """Все записи очереди (ожидающие и «поверх») с живыми процессами; мёртвые убираются."""
    qd = queue_dir(lock)
    out = []
    try:
        files = sorted(qd.glob("*.json"))
    except OSError:
        return out
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue                                   # пишется прямо сейчас - прочитаем в следующий раз
        if not alive(d.get("pid")):
            try:
                f.unlink()
            except OSError:
                pass
            continue
        since = dt.datetime.fromisoformat(d["since"]) if d.get("since") else None
        out.append(Holder(program=d.get("program") or "?", pid=d.get("pid"), what=d.get("what", ""),
                          who=d.get("who", HUMAN), since=since, kind=d.get("kind", "wait"),
                          ts=float(d.get("ts", 0)), token=d.get("token", ""), file=f))
    return out


def _order(h: Holder) -> tuple:
    return (0 if h.who == HUMAN else 1, h.ts, h.file.name if h.file else "")


def queue(path: Path | None = None) -> list[Holder]:
    """Ожидающие в порядке очереди: люди по времени, потом фон по времени (README, 5.2)."""
    lock = Path(path) if path else default_path()
    return sorted((h for h in _entries(lock) if h.kind == "wait"), key=_order)


def alongside(path: Path | None = None) -> list[Holder]:
    """Кто работает поверх чужого замка («Запустить сейчас»)."""
    lock = Path(path) if path else default_path()
    return [h for h in _entries(lock) if h.kind == "now"]


def status_json(path: Path | None = None, me: "HeavyLock | None" = None) -> dict:
    """Для веб-окна: GET /api/heavy-lock."""
    lock = Path(path) if path else (me.path if me else default_path())
    h = status(lock)
    q = queue(lock)
    d = {"free": h is None, "holder": h.to_json() if h else None,
         "queue": [w.to_json() for w in q], "alongside": [w.to_json() for w in alongside(lock)],
         "version": VERSION}
    if me is not None:
        d["me"] = {"held": me.held, "waiting": me.waiting, "now": me.now, "ahead": me.ahead() if me.waiting else 0,
                   "idle_released": me.idle_released, "program": me.program}
    return d


# ============================================================ журнал
def journal(lock: Path, event: str, program: str, pid: int | None, who: str = "", what: str = "",
            held: float | None = None, note: str = "") -> None:
    """Строка журнала: время, событие, программа, процесс, кто, что, сколько держал (с), заметка."""
    qd = queue_dir(lock)
    try:
        qd.mkdir(exist_ok=True)       # рядом с замком; чужих папок не создаём
        f = qd / "journal.log"
        line = "\t".join([dt.datetime.now().isoformat(timespec="seconds"), event, program, str(pid or ""), who,
                          what.replace("\t", " ").replace("\n", " "),
                          "" if held is None else f"{held:.0f}", note.replace("\t", " ")]) + "\n"
        with open(f, "a", encoding="utf-8") as fh:
            fh.write(line)
        if f.stat().st_size > JOURNAL_MAX:
            data = f.read_text(encoding="utf-8", errors="ignore")
            f.write_text(data[len(data) // 2:].split("\n", 1)[-1], encoding="utf-8")
    except OSError:
        pass


# ============================================================ замок
def _slug(s: str) -> str:
    return re.sub(r"[^\w]+", "_", s).strip("_")[:40] or "program"


class HeavyLock:
    """Замок видеокарты одной операции программы: try_acquire / acquire (очередь) / run_now / release,
    touch / idle_check / resume (простой человека), checkpoint (фон уступает человеку)."""

    def __init__(self, program: str, what: str = "", who: str | None = None, path: Path | str | None = None,
                 idle_after: float | None = IDLE_AFTER, held_env: str | None = None, path_env: str | None = None,
                 clock=time.monotonic, pid: int | None = None) -> None:
        self.program = program
        self.what = what
        env_who = os.environ.get("HEAVY_LOCK_WHO")
        self.who = who or (env_who if env_who in (HUMAN, BACKGROUND) else HUMAN)
        self._path = Path(path) if path else None
        self._path_env = path_env
        self.idle_after = idle_after
        self.held_env = held_env
        self.clock = clock
        self.pid = pid or os.getpid()
        self.token = uuid.uuid4().hex
        self.held = False           # замок наш (или его держит тот, кто нас запустил, или замка нет вовсе)
        self.outside = False        # держит запустивший (HEAVY_LOCK_HELD=1)
        self.unavailable = ""       # папки нет или нет прав - работаем без замка (README, 5.8)
        self.now = False            # работаем поверх чужого замка
        self.waiting = False
        self.idle_released = False
        self.last_active = clock()
        self._t_take = 0.0
        self._ts = 0.0               # когда встали в очередь
        self._qfile: Path | None = None
        self._nowfile: Path | None = None
        self._mu = threading.RLock()

    # ---------- общее
    @property
    def path(self) -> Path:
        return self._path or default_path(self._path_env)

    def held_outside(self) -> bool:
        return os.environ.get("HEAVY_LOCK_HELD") == "1" or bool(self.held_env and os.environ.get(self.held_env) == "1")

    def holder(self) -> Holder | None:
        return status(self.path)

    def ours(self) -> bool:
        """Замок у этого объекта (или у запустившего)."""
        return self.held

    def _journal(self, event: str, held: float | None = None, note: str = "") -> None:
        journal(self.path, event, self.program, self.pid, self.who, self.what, held, note)

    def _entry(self, kind: str) -> Path | None:
        qd = queue_dir(self.path)
        now = dt.datetime.now()
        ts = time.time()
        name = f"{'now-' if kind == 'now' else ''}{now:%Y%m%d-%H%M%S}-{now:%f}-{_slug(self.program)}-{self.pid}.json"
        d = {"program": self.program, "pid": self.pid, "what": self.what, "who": self.who, "kind": kind,
             "since": now.isoformat(timespec="seconds"), "ts": ts, "token": self.token, "version": VERSION}
        try:
            qd.mkdir(exist_ok=True)       # рядом с замком; чужих папок не создаём
            tmp = qd / (name + ".tmp")
            tmp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, qd / name)                  # целиком или никак - читатель не увидит полфайла
            return qd / name
        except OSError:
            return None

    @staticmethod
    def _drop(f: Path | None) -> None:
        if f is not None:
            try:
                f.unlink()
            except OSError:
                pass

    # ---------- очередь
    def ahead(self) -> int:
        """Сколько живых ожидающих передо мной. Не в очереди - как будто встал в конец своего класса."""
        q = [h for h in queue(self.path) if h.token != self.token]
        if self._qfile is not None:
            mine = Holder(self.program, self.pid, who=self.who, ts=self._ts, file=self._qfile)
            return sum(1 for h in q if _order(h) < _order(mine))
        if self.who == HUMAN:
            return sum(1 for h in q if h.who == HUMAN)
        return len(q)

    def ahead_names(self) -> list[str]:
        q = [h for h in queue(self.path) if h.token != self.token]
        return [h.program for h in q][: max(0, self.ahead())]

    # ---------- взять / отдать
    def try_acquire(self) -> bool:
        """Взять, если свободно и передо мной в очереди никого. Не ждёт."""
        with self._mu:
            if self.held:
                return True
            if self.held_outside():
                self.held = self.outside = True
                return True
            lock = self.path
            if lock.exists() and not lock.is_dir():
                self.unavailable = f"{lock} - не папка"
                self.held = True
                return True
            if self.ahead() > 0:
                return False
            for _ in range(3):              # повтор - после снятия брошенного замка
                try:
                    lock.mkdir()
                except FileExistsError:
                    if self._break_stale(lock):
                        continue
                    return False
                except OSError as e:        # нет папки проектов или прав - работаем без замка
                    self.unavailable = str(e)
                    self.held = True
                    return True
                try:
                    self._write_owner(lock)
                except OSError as e:
                    shutil.rmtree(lock, ignore_errors=True)
                    self.unavailable = str(e)
                    self.held = True
                    return True
                self.held = True
                self.idle_released = False
                if self.now:                    # работали поверх - теперь замок наш
                    self.now = False
                    self._drop(self._nowfile)
                    self._nowfile = None
                self._t_take = time.time()
                self.last_active = self.clock()
                self._drop(self._qfile)
                self._qfile = None
                self._journal("take")
                return True
            return False

    def _write_owner(self, lock: Path) -> None:
        now = dt.datetime.now()
        (lock / "owner").write_text(f"{self.program} {now:%a %b %d %H:%M:%S %Y} pid {self.pid}\n", encoding="utf-8")
        d = {"program": self.program, "pid": self.pid, "what": self.what, "who": self.who,
             "since": now.isoformat(timespec="seconds"), "host": socket.gethostname(), "token": self.token,
             "version": VERSION}
        tmp = lock / "owner.json.tmp"
        tmp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, lock / "owner.json")

    def _break_stale(self, lock: Path) -> bool:
        """Снять брошенный замок (процесс владельца мёртв). Переименованием - чтобы не удалить свежий чужой."""
        h = _read_holder(lock)
        if h is None:
            age = _age(lock)
            if age is None:
                return True                 # папку уже убрали - пробуем взять снова
            if age < WRITING_GRACE or not _stale(lock, None):
                return False
        elif not _stale(lock, h):
            return False
        tmp = lock.with_name(f"{lock.name}.stale-{self.pid}-{uuid.uuid4().hex[:6]}")
        try:
            os.rename(lock, tmp)
        except OSError:
            return True                     # кто-то снял раньше нас
        h2 = _read_holder(tmp)
        if h is not None and h2 is not None and (h2.pid, h2.token, h2.since) != (h.pid, h.token, h.since):
            try:                            # пока проверяли, замок сменился на живой - вернуть на место
                os.rename(tmp, lock)
            except OSError:
                pass
            return False
        shutil.rmtree(tmp, ignore_errors=True)
        journal(lock, "stale", h.program if h else "?", h.pid if h else None, h.who if h else "",
                h.what if h else "", note=f"снял {self.program}")
        return True

    def acquire(self, cancel: threading.Event | None = None, on_wait=None, wait: float | None = None,
                poll: float = 1.0) -> bool:
        """«Подождать очереди»: встать в очередь и ждать. on_wait(holder, ahead) - раз в poll с.
        False - отмена (cancel) или вышло время wait."""
        if self.try_acquire():
            return True
        with self._mu:
            self._ts = time.time()
            self._qfile = self._entry("wait")
            self.waiting = True
        self._journal("wait", note=(self.holder() or Holder("-")).program)
        t0 = self.clock()
        try:
            while True:
                if cancel is not None and cancel.is_set():
                    self._journal("cancel")
                    return False
                if self._qfile is not None and not self._qfile.exists():
                    self._qfile = self._entry("wait")     # файл очереди пропал (чистка) - встать заново
                if self.try_acquire():
                    return True
                if wait is not None and self.clock() - t0 > wait:
                    self._journal("timeout")
                    return False
                if on_wait is not None:
                    on_wait(self.holder(), self.ahead())
                if cancel is not None:
                    cancel.wait(poll)
                else:
                    time.sleep(poll)
        finally:
            with self._mu:
                self.waiting = False
                if not self.held:
                    self._drop(self._qfile)
                    self._qfile = None

    def run_now(self) -> None:
        """«Запустить сейчас»: работать поверх чужого замка (README, 5.4)."""
        with self._mu:
            if self.held or self.now:
                return
            self.now = True
            self._nowfile = self._entry("now")
            self.last_active = self.clock()
        self._journal("now", note=(self.holder() or Holder("-")).program)

    def release(self, reason: str = "release") -> None:
        with self._mu:
            if self.now:
                self.now = False
                self._drop(self._nowfile)
                self._nowfile = None
                self._journal("now-end")
            if not self.held:
                return
            self.held = False
            if self.outside or self.unavailable:
                self.outside, self.unavailable = False, ""
                return
            lock = self.path
            h = _read_holder(lock)
            if h is not None and h.pid == self.pid and (h.legacy or h.token == self.token):
                shutil.rmtree(lock, ignore_errors=True)
            self._journal(reason, held=time.time() - self._t_take if self._t_take else None)

    # ---------- простой человека (README, 5.5)
    def touch(self) -> None:
        self.last_active = self.clock()

    def idle_check(self) -> bool:
        """Вызывать раз в несколько секунд. True - замок отпущен по простою."""
        if (self.held and not self.outside and not self.unavailable and self.who == HUMAN and self.idle_after
                and self.clock() - self.last_active >= self.idle_after):
            self.release(reason="idle")
            self.idle_released = True
            return True
        return False

    def resume(self) -> bool:
        """Продолжение после простоя: взять снова. False - занято, показать окно «Видеокарта занята»."""
        if self.held:
            self.touch()
            return True
        if self.try_acquire():
            self.idle_released = False
            self.touch()
            return True
        return False

    # ---------- фон уступает человеку (README, 5.3)
    def human_waiting(self) -> bool:
        return any(h.who == HUMAN and h.token != self.token for h in queue(self.path))

    def checkpoint(self, cancel: threading.Event | None = None, on_wait=None, poll: float = 1.0) -> bool:
        """Контрольная точка фона: ждёт человек - отдать, встать в очередь, получить снова.
        True - уступили (после - self.held: получили ли обратно; False - отмена)."""
        if not self.held or self.outside or self.unavailable or self.who != BACKGROUND or not self.human_waiting():
            return False
        self.release(reason="yield")
        self.acquire(cancel=cancel, on_wait=on_wait, poll=poll)
        return True

    def set_what(self, what: str) -> None:
        self.what = what
        with self._mu:
            if self.held and not self.outside and not self.unavailable:
                try:
                    self._write_owner(self.path)
                except OSError:
                    pass

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()


# ============================================================ командная строка (heavy_lock.sh)
def _describe(h: Holder) -> str:
    who = "фон" if h.who == BACKGROUND else "человек"
    what = f", {h.what}" if h.what else ""
    return f"«{h.program}», с {h.since_text}{what} ({who}, pid {h.pid or '?'})"


def _cli_status(lock: Path, as_json: bool) -> int:
    if as_json:
        print(json.dumps(status_json(lock), ensure_ascii=False, indent=1))
        return 0
    h = status(lock)
    print("Видеокарта свободна." if h is None else "Видеокарта занята: " + _describe(h))
    for i, w in enumerate(queue(lock), 1):
        print(f"  в очереди {i}: {_describe(w)}")
    for w in alongside(lock):
        print(f"  работает поверх: {_describe(w)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="heavy_lock.sh", description="Очередь видеокарты .heavy-lock")
    ap.add_argument("--path", help="путь к .heavy-lock (иначе HEAVY_LOCK или «Разработка ПО»)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("take", "release", "yield", "run"):
        p = sub.add_parser(name)
        p.add_argument("program")
        if name in ("take", "run"):
            p.add_argument("what", nargs="?", default="")
        p.add_argument("--who", choices=(HUMAN, BACKGROUND), default=BACKGROUND)
        p.add_argument("--pid", type=int, default=None, help="процесс-владелец (по умолчанию - вызвавший)")
        p.add_argument("--wait-min", type=float, default=330)
        if name == "run":
            p.add_argument("command", nargs=argparse.REMAINDER)
    st = sub.add_parser("status")
    st.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    lock = Path(a.path) if a.path else default_path()
    if a.cmd == "status":
        return _cli_status(lock, a.json)

    def log(msg: str) -> None:
        print(f"{dt.datetime.now():%H:%M:%S} {msg}", flush=True)

    shown = {"t": 0.0}

    def waiting(h: Holder | None, ahead: int) -> None:
        if time.time() - shown["t"] >= 60:          # раз в минуту, как прежние скрипты
            shown["t"] = time.time()
            who = _describe(h) if h else "свободна, но перед нами очередь"
            log(f"жду видеокарту: {who}; перед нами: {ahead}")

    if a.cmd == "run":
        cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
        if not cmd:
            ap.error("run: нужна команда после --")
        lk = HeavyLock(a.program, a.what, who=a.who, path=lock, idle_after=None)
        if not lk.acquire(on_wait=waiting, wait=a.wait_min * 60, poll=5):
            log(f"видеокарту не получил за {a.wait_min:g} мин (занята: {_describe(lk.holder()) if lk.holder() else '?'})")
            return 75
        log("видеокарта взята")
        env = dict(os.environ, HEAVY_LOCK_HELD="1", HEAVY_LOCK=str(lock))
        proc = None

        def stop(sig, _frm):
            if proc is not None and proc.poll() is None:
                proc.send_signal(sig)
        for s in (signal.SIGTERM, signal.SIGINT):
            signal.signal(s, stop)
        try:
            proc = subprocess.Popen(cmd, env=env)
            return proc.wait()
        finally:
            lk.release()
            log("видеокарта отдана")

    pid = a.pid or os.getppid()
    lk = HeavyLock(a.program, getattr(a, "what", ""), who=a.who, path=lock, idle_after=None, pid=pid)
    if a.cmd == "take":
        if not lk.acquire(on_wait=waiting, wait=a.wait_min * 60, poll=5):
            log(f"видеокарту не получил за {a.wait_min:g} мин")
            return 75
        log(f"видеокарта взята: {a.program}, pid {pid}")
        return 0
    h = _read_holder(lock) if lock.is_dir() else None
    mine = h is not None and h.pid == pid
    if a.cmd == "release":
        if not mine:
            log(f"замок не наш ({_describe(h) if h else 'свободен'}) - не трогаю")
            return 0
        shutil.rmtree(lock, ignore_errors=True)
        journal(lock, "release", a.program, pid, a.who,
                held=(dt.datetime.now() - h.since).total_seconds() if h.since else None)
        log("видеокарта отдана")
        return 0
    # yield: контрольная точка фона
    if not mine or not any(w.who == HUMAN for w in queue(lock)):
        return 0
    what = h.what
    shutil.rmtree(lock, ignore_errors=True)
    journal(lock, "yield", a.program, pid, a.who, what)
    log("ждёт человек - видеокарта отдана, встаю в очередь")
    lk.what = what
    if not lk.acquire(on_wait=waiting, wait=a.wait_min * 60, poll=5):
        return 75
    log("видеокарта снова взята")
    return 0


if __name__ == "__main__":
    sys.exit(main())
