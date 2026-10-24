/* Окно «Видеокарта занята» для веб-программ (IC, CD, LKD, LDB) - README, 6.3.
   Тексты - ../texts.json (тот же файл, что у Python). Сервер отдаёт GET /api/heavy-lock = heavy_lock.status_json().

   HeavyLockUI.ask(holder, {lang, cpu, queue}) → Promise<"wait" | "now" | "cpu" | null>
   cpu: true - кнопка «На процессоре»; {seconds, factor} - с прогнозом (> 3 раз медленнее - «очень медленно»)
   HeavyLockUI.describe(holder, lang), HeavyLockUI.waitingLine(holder, ahead, lang)
   HeavyLockUI.headerState(status, lang, {modelOk, working, error}) → {state, text, kind}  (стандарт 2.1) */
(function (g) {
  "use strict";
  const base = new URL(".", document.currentScript ? document.currentScript.src : location.href);
  let TEXTS = null;
  const ready = fetch(new URL("../texts.json", base)).then((r) => r.json()).then((d) => { TEXTS = d; return d; });
  const KIND = { ready: "ok", working: "acc", busy: "warn", busy_queued: "warn", alongside: "warn",
    nomodel: "neutral", error: "bad" };

  function t(key, lang, kw) {
    const L = (TEXTS && (TEXTS[lang] || TEXTS.ru)) || {};
    let s = L[key] || (TEXTS && TEXTS.ru[key]) || key;
    for (const [k, v] of Object.entries(kw || {})) s = s.split("{" + k + "}").join(v);
    return s;
  }
  function program(h, lang) { return !h || !h.program || h.program === "?" ? t("someone", lang) : h.program; }
  function describe(h, lang) {
    const kw = { program: program(h, lang), since: (h && h.since_text) || "?", what: (h && h.what) || "" };
    return t(h && h.what ? "busy" : "busy_nowhat", lang, kw);
  }
  function waitingLine(h, ahead, lang) {
    const head = !h ? t("waiting_queue", lang)
      : t(h.what ? "waiting" : "waiting_nowhat", lang, { program: program(h, lang), since: h.since_text, what: h.what });
    return head + " · " + (ahead > 0 ? t("waiting_ahead", lang, { n: ahead }) : t("waiting_next", lang));
  }
  const SLOW_FACTOR = 3;
  function etaText(sec, lang) {
    const s = Math.max(0, Math.round(sec));
    if (s < 60) return t("eta_sec", lang, { n: Math.max(5, Math.round(s / 5) * 5) });
    let m = Math.round(s / 60);
    if (m < 60) return t("eta_min", lang, { n: m });
    const h = Math.floor(m / 60); m %= 60;
    return m ? t("eta_h_min", lang, { h, n: m }) : t("eta_h", lang, { h });
  }
  function cpuOption(cpu, lang) {
    if (!cpu) return null;
    if (cpu === true) return { label: t("cpu", lang), hint: t("cpu_hint", lang), slow: false };
    const eta = cpu.seconds ? etaText(cpu.seconds, lang) : "";
    const x = Number(cpu.factor || 1);
    if (x > SLOW_FACTOR) return { label: t("cpu_slow_btn", lang), hint: t("cpu_slow", lang, { eta: eta || "?", x: x.toFixed(0) }), slow: true };
    return { label: t("cpu", lang), hint: eta ? t("cpu_eta", lang, { eta }) : t("cpu_hint", lang), slow: false };
  }
  function headerState(st, lang, o) {
    o = o || {};
    const me = (st && st.me) || {};
    let s;
    if (o.error) s = "error";
    else if (me.now) return { state: "alongside", text: t("st_alongside", lang, { program: program(st.holder, lang) }), kind: KIND.alongside };
    else if (o.working || me.held) s = "working";
    else if (st && (st.holder || me.waiting)) {
      s = me.waiting ? "busy_queued" : "busy";
      return { state: s, text: t("st_" + s, lang, { program: program(st.holder, lang) }), kind: KIND[s] };
    } else s = o.modelOk === false ? "nomodel" : "ready";
    return { state: s, text: t("st_" + s, lang), kind: KIND[s] };
  }

  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  // открытый модальный <dialog> программы - в верхнем слое страницы: окно очереди - внутри него, иначе его не нажать
  function topLayer() {
    const open = [...document.querySelectorAll("dialog[open]")].filter((d) => {
      try { return d.matches(":modal"); } catch (e) { return true; }
    });
    return open.length ? open[open.length - 1] : document.body;
  }

  async function ask(holder, opts) {
    opts = opts || {};
    const lang = opts.lang || document.documentElement.lang || "ru";
    await ready;
    return new Promise((resolve) => {
      const back = el("div", "hl-backdrop");
      const box = el("div", "hl-dialog");
      box.setAttribute("role", "dialog");
      box.setAttribute("aria-modal", "true");
      box.setAttribute("aria-labelledby", "hl-title");
      const h = el("h2", "hl-title", describe(holder, lang)); h.id = "hl-title";   // заголовок - сама строка «занята: …»
      box.append(h);
      if (holder && holder.who === "background") box.append(el("p", "hl-note", t("busy_bg", lang)));
      if (opts.queue && opts.queue.length) {
        box.append(el("p", "hl-note", t("queue_ahead", lang, { names: opts.queue.map((n) => "«" + n + "»").join(", ") })));
      }
      const rows = [["wait", "hl-primary", t("wait", lang), t("wait_hint", lang), ""],
        ["now", "", t("now", lang), t("now_hint", lang), "hl-warn"]];
      const co = cpuOption(opts.cpu, lang);
      if (co) rows.push(["cpu", co.slow ? "hl-ghost hl-slow" : "", co.label, co.hint, co.slow ? "hl-warn" : ""]);
      const list = el("div", "hl-options");
      const done = (ans) => { document.removeEventListener("keydown", onKey, true); back.remove(); resolve(ans); };
      for (const [key, cls, label, hint, hcls] of rows) {
        const b = el("button", "hl-btn " + cls, label);
        b.type = "button"; b.dataset.answer = key;
        b.addEventListener("click", () => done(key));
        list.append(b, el("span", "hl-hint " + hcls, hint));
      }
      const foot = el("div", "hl-foot");
      const cancel = el("button", "hl-btn hl-ghost", t("cancel", lang));
      cancel.type = "button"; cancel.dataset.answer = "cancel";
      cancel.addEventListener("click", () => done(null));
      foot.append(cancel);
      box.append(list, foot);
      back.append(box);
      back.addEventListener("click", (e) => { if (e.target === back) done(null); });
      function onKey(e) {
        if (e.key === "Escape") { e.preventDefault(); done(null); }
        else if (e.key === "Enter" && !(e.target instanceof HTMLButtonElement)) { e.preventDefault(); done("wait"); }
      }
      document.addEventListener("keydown", onKey, true);
      topLayer().append(back);
      list.querySelector("button").focus();
    });
  }

  g.HeavyLockUI = { ready, t, describe, waitingLine, headerState, ask, cpuOption, etaText };
})(window);
