# -*- coding: utf-8 -*-
"""Общий блок ci (README, раздел 4). Своя временная папка: проект-пробник с git и «GitHub» - пустой репозиторий рядом.

    python3 "_общие модули/ci/tests/test_ci.py"
"""
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CI = Path(__file__).resolve().parents[1]
HAS_RUFF = bool(importlib.util.find_spec("ruff"))
# Python пробника - этот же; нет ruff - шаг ruff заглушкой (его проверяет test_ruff_finds_undefined_name)
BASE = f'PY="{sys.executable}"\nRUFF_PATHS=(src)\n' + ("" if HAS_RUFF else 'ruff_step() { echo "без ruff"; }\n')
GOOD = BASE + 'unit() { echo unit-ok; }\nrepo() { echo repo-ok; }\n'
BAD = BASE + 'unit() { echo unit-fail; return 3; }\nrepo() { echo repo-ok; }\n'


def run(args, cwd, **kw):
    env = {k: v for k, v in os.environ.items() if k != "CI"}
    env.update(GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t",
               **kw.pop("env", {}))
    return subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, **kw)


class CheckAll(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ci-test-"))
        self.p = self.tmp / "Проект с пробелом"
        (self.p / "src").mkdir(parents=True)
        (self.p / "src" / "a.py").write_text("x = 1\nprint(x)\n", encoding="utf-8")
        shutil.copytree(CI, self.p / "common" / "ci", ignore=shutil.ignore_patterns("tests", "__pycache__"))
        (self.p / "scripts").mkdir()
        (self.p / "scripts" / "check_all.conf").write_text(GOOD, encoding="utf-8")
        self.remote = self.tmp / "remote.git"
        run(["git", "init", "-q", "--bare", str(self.remote)], self.tmp)
        run(["git", "init", "-q", "-b", "main"], self.p)
        run(["git", "remote", "add", "origin", str(self.remote)], self.p)
        r = run(["bash", "common/ci/install.sh"], self.p)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        run(["git", "add", "."], self.p)
        run(["git", "commit", "-qm", "init"], self.p)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def check(self, *args, **kw):
        return run(["scripts/check_all.sh", *args], self.p, **kw)

    def test_all_pass(self):
        r = self.check()
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("unit-ok", r.stdout)
        self.assertIn("unit - да", r.stdout)
        self.assertIn("repo - да", r.stdout)
        self.assertNotIn("ui -", r.stdout)                      # окно - только по --ui

    def test_unit_fails(self):
        (self.p / "scripts" / "check_all.conf").write_text(BAD, encoding="utf-8")
        r = self.check()
        self.assertEqual(r.returncode, 1)
        self.assertIn("unit - НЕТ (код 3)", r.stdout)
        self.assertIn("repo - да", r.stdout)                    # остальные шаги всё равно идут

    @unittest.skipUnless(HAS_RUFF, "нет ruff")
    def test_ruff_finds_undefined_name(self):
        (self.p / "src" / "b.py").write_text("print(nope)\n", encoding="utf-8")
        r = self.check("--only", "ruff")
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("F821", r.stdout)

    def test_ui_by_flag(self):
        r = self.check("--ui")
        self.assertEqual(r.returncode, 1)                       # окно не задано в check_all.conf
        self.assertIn("ui - НЕТ", r.stdout)

    def test_bad_key(self):
        self.assertEqual(self.check("--fast").returncode, 2)

    def test_ci_uses_runner_python(self):
        (self.p / "scripts" / "check_all.conf").write_text('unit() { echo "py=$PY"; }\n', encoding="utf-8")   # без своего PY
        self.assertIn("py=python\n", self.check("--only", "unit", env={"CI": "true"}).stdout)
        (self.p / ".venv" / "bin").mkdir(parents=True)
        (self.p / ".venv" / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
        (self.p / ".venv" / "bin" / "python").chmod(0o755)
        self.assertIn("/.venv/bin/python", self.check("--only", "unit").stdout)

    def test_workflow_drift_warning(self):
        run(["bash", "common/ci/install.sh", "--workflow"], self.p)
        self.assertTrue(filecmp_same(self.p / ".github/workflows/linux.yml", CI / "linux.yml"))
        self.assertNotIn("отличается", self.check().stdout)
        with open(self.p / ".github/workflows/linux.yml", "a", encoding="utf-8") as f:
            f.write("# правка\n")
        r = self.check()
        self.assertEqual(r.returncode, 0)                       # предупреждение, не ошибка
        self.assertIn("отличается от common/ci/linux.yml", r.stdout)

    def test_hook_blocks_bad_push(self):
        (self.p / "scripts" / "check_all.conf").write_text(BAD, encoding="utf-8")
        r = run(["git", "push", "-q", "origin", "main"], self.p)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("unit-fail", r.stderr)
        self.assertNotIn("main", run(["git", "branch", "--list"], self.remote).stdout)
        r = run(["git", "push", "-q", "--no-verify", "origin", "main"], self.p)  # осознанный пропуск
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_hook_lets_good_push(self):
        r = run(["git", "push", "-q", "origin", "main"], self.p)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("pre-push: проверка", r.stderr)
        self.assertIn("main", run(["git", "branch", "--list"], self.remote).stdout)

    def test_delete_not_checked(self):
        run(["git", "push", "-q", "origin", "main"], self.p)
        run(["git", "tag", "v0"], self.p)
        run(["git", "push", "-q", "origin", "v0"], self.p)
        (self.p / "scripts" / "check_all.conf").write_text(BAD, encoding="utf-8")
        r = run(["git", "push", "-q", "origin", ":refs/tags/v0"], self.p)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("pre-push: проверка", r.stderr)

    def test_install_keeps_foreign_hook(self):
        hooks = self.p / ".git" / "hooks"
        (hooks / "pre-push").write_text("#!/bin/sh\necho чужой\n", encoding="utf-8")
        run(["bash", "common/ci/install.sh"], self.p)
        self.assertIn("чужой", (hooks / "pre-push.old").read_text(encoding="utf-8"))
        self.assertIn("common/ci/pre-push", (hooks / "pre-push").read_text(encoding="utf-8"))
        before = (self.p / "scripts" / "check_all.sh").read_text(encoding="utf-8")
        run(["bash", "common/ci/install.sh"], self.p)                       # второй раз - ничего не меняется
        self.assertEqual((self.p / "scripts" / "check_all.sh").read_text(encoding="utf-8"), before)
        self.assertIn("чужой", (hooks / "pre-push.old").read_text(encoding="utf-8"))


def filecmp_same(a, b):
    return Path(a).read_bytes() == Path(b).read_bytes()


if __name__ == "__main__":
    unittest.main()
