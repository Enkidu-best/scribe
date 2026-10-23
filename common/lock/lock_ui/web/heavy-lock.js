/* Окно «Видеокарта занята» для веб-программ (IC, CD, LKD, LDB) - README, 6.3.
   Тексты - ../texts.json (тот же файл, что у Python). Сервер отдаёт GET /api/heavy-lock = heavy_lock.status_json().

   HeavyLockUI.ask(holder, {lang, cpu, queue}) → Promise<"wait" | "now" | "cpu" | null>
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
      const rows = [["wait", "hl-primary", "wait_hint", ""], ["now", "", "now_hint", "hl-warn"]];
      if (opts.cpu) rows.push(["cpu", "", "cpu_hint", ""]);
      const list = el("div", "hl-options");
      const done = (ans) => { document.removeEventListener("keydown", onKey, true); back.remove(); resolve(ans); };
      for (const [key, cls, hint, hcls] of rows) {
        const b = el("button", "hl-btn " + cls, t(key, lang));
        b.type = "button"; b.dataset.answer = key;
        b.addEventListener("click", () => done(key));
        list.append(b, el("span", "hl-hint " + hcls, t(hint, lang)));
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
      document.body.append(back);
      list.querySelector("button").focus();
    });
  }

  g.HeavyLockUI = { ready, t, describe, waitingLine, headerState, ask };
})(window);
