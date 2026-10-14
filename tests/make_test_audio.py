#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сборка тестового набора для режима «Несколько» (RU/EN) и нормализации - Scribe 2.2.

Голоса синтеза - системные (`say`): Milena (ru), Samantha / Daniel (en). Для
каждой записи рядом пишется JSON с эталоном: время, язык и текст каждой фразы.
Аудио в git не кладём - только этот скрипт. Запуск:

    python3 tests/make_test_audio.py            # всё в tests/audio/

Записи:
  lesson.m4a        урок: ru и en чередуются через 1-3 предложения, паузы 0,2-1,5 с
  long_insert.m4a   ~10 мин русской речи, в середине английская вставка ~40 с
  mix.m4a           смесь в одном предложении: «давай повторим слово apple» (слово - другим голосом)
  mix_same.m4a      та же смесь целиком одним голосом
  quiet.m4a         русская речь на уровне около -35 dB (для проверки нормализации)
  dictation.m4a     ~5 мин для диктофона: с паузами, без пауз, короткое «Спасибо.»
  english.m4a       чисто английская запись (~4 мин) - «Авто» не должен счесть её смешанной
  english_accent.m4a то же, прочитано русским голосом (английский с русским акцентом)
"""
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SR = 16000
OUT = Path(__file__).with_name("audio")
RU_VOICE = "Milena"
EN_VOICES = ["Samantha", "Daniel"]

# --- урок: объяснения по-русски, примеры и реплики по-английски ---
LESSON_RU = [
    "Добрый день, начинаем наше занятие.",
    "Сегодня мы разберём настоящее совершённое время.",
    "Сначала послушайте пример и повторите за мной.",
    "Обратите внимание на вспомогательный глагол.",
    "Это время связывает прошлое с настоящим моментом.",
    "Теперь попробуйте составить своё предложение.",
    "Хорошо, а теперь переведите на английский.",
    "Почти правильно, но порядок слов другой.",
    "Давайте вспомним неправильные глаголы.",
    "Третья форма глагола здесь обязательна.",
    "Не торопитесь, подумайте ещё немного.",
    "Отлично, вы уже говорите гораздо увереннее.",
    "Запишите это правило в тетрадь.",
    "Сравните два предложения на экране.",
    "В первом случае действие уже закончилось.",
    "Во втором случае мы говорим о привычке.",
    "Теперь прочитайте диалог по ролям.",
    "Вы будете клиентом, а я буду менеджером.",
    "Следите за интонацией в вопросах.",
    "Вопрос начинается с вспомогательного глагола.",
    "Хорошо, переходим к следующему упражнению.",
    "Откройте учебник на странице с текстом про путешествия.",
    "Найдите в тексте все примеры нового времени.",
    "Сколько примеров вы нашли?",
    "Правильно, их действительно пять.",
    "А теперь расскажите о своём последнем отпуске.",
    "Используйте прошедшее простое время.",
    "Это частая ошибка, давайте её разберём.",
    "Слово уже обычно стоит перед смысловым глаголом.",
    "Повторите, пожалуйста, ещё раз медленно.",
    "Замечательно, произношение стало намного лучше.",
    "На дом прочитайте следующий раздел.",
    "Выпишите незнакомые слова и найдите их значение.",
    "В следующий раз начнём с короткой проверки.",
    "Есть ли у вас вопросы по сегодняшней теме?",
    "Тогда на этом заканчиваем, спасибо за работу.",
]
LESSON_EN = [
    "I have already finished my homework.",
    "She has never been to London.",
    "Have you ever tried Japanese food?",
    "We have lived here for five years.",
    "They have just arrived at the airport.",
    "Could you repeat the question, please?",
    "I think the answer is in the second paragraph.",
    "Let me check the list of irregular verbs.",
    "My brother has lost his keys again.",
    "How long have you worked for this company?",
    "I have been a lawyer since two thousand fifteen.",
    "Good morning, how can I help you today?",
    "I would like to book a table for two.",
    "Is breakfast included in the price?",
    "Our flight has been delayed by two hours.",
    "I usually take the train to work.",
    "Last summer we went to the seaside.",
    "The weather was perfect and the hotel was quiet.",
    "We visited three museums and an old castle.",
    "I have read this book twice.",
    "He has already sent the contract to the client.",
    "Has the meeting started yet?",
    "Not yet, they are still waiting for the manager.",
    "I have not seen him since Monday.",
    "This is the best coffee I have ever had.",
    "Please open your books at page twenty.",
    "What does this word mean?",
    "It means a short trip for pleasure.",
    "Sorry, I did not catch that.",
    "Can you spell your last name, please?",
    "We have made a lot of progress this month.",
    "Thank you, that was very helpful.",
]

# --- английская вставка для длинной записи (~40 с) ---
INSERT_EN = [
    "Now let me switch to English for a moment, because this part was prepared for our international colleagues.",
    "Over the last quarter we have reviewed more than three hundred contracts and reduced the average approval time by almost a week.",
    "The main reason is the new checklist, which helps lawyers find risky clauses before the business sends the draft to the counterparty.",
    "Next month we plan to train the sales team, so that simple questions no longer reach the legal department at all.",
    "We will also publish a short guide with answers to the most common questions about signatures and powers of attorney.",
    "If you have suggestions for the guide, please send them to me by the end of next week.",
    "That is all from my side, and now I will continue in Russian.",
]

# --- русская речь для длинной и тихой записи: деловое совещание ---
LONG_RU = [
    "Коллеги, давайте начнём совещание, у нас сегодня довольно плотная повестка.",
    "Первый вопрос касается подготовки к проверке, которая начнётся в следующем месяце.",
    "Нам нужно заранее собрать все договоры с крупными поставщиками и проверить сроки их действия.",
    "Часть документов хранится в электронном архиве, а часть всё ещё лежит в бумажном виде.",
    "Предлагаю поручить сверку отделу документооборота, у них есть доступ к обоим архивам.",
    "Второй вопрос связан с новым регламентом согласования сделок.",
    "Проект регламента разослан всем руководителям подразделений, но замечания прислали не все.",
    "Самые частые замечания касаются сроков согласования и порядка замены отсутствующих сотрудников.",
    "Мы учли большую часть предложений и подготовили обновлённую редакцию документа.",
    "Отдельно обсудим вопрос об электронной подписи для руководителей филиалов.",
    "Сейчас документы подписываются вручную, и это занимает до трёх рабочих дней.",
    "С электронной подписью процесс займёт несколько часов, если все участники на месте.",
    "При этом нужно продумать хранение ключей и порядок выдачи доверенностей.",
    "Третий вопрос касается обучения новых сотрудников юридического департамента.",
    "За последние полгода к нам пришли четыре человека, и им не хватает практических материалов.",
    "Предлагаю собрать короткую памятку с типовыми ситуациями и ответами на них.",
    "Также полезно назначить каждому новичку наставника на первые два месяца.",
    "Наставник будет проверять первые заключения и помогать с внутренними процедурами.",
    "Перейдём к претензионной работе, здесь есть несколько важных новостей.",
    "Количество претензий от покупателей снизилось по сравнению с прошлым годом.",
    "Однако выросло число споров с подрядчиками по срокам выполнения работ.",
    "Мы проанализировали причины и видим, что многие проблемы заложены ещё в договорах.",
    "В договорах часто нет чёткого графика работ и порядка приёмки результата.",
    "Поэтому предлагаю обновить типовую форму договора подряда.",
    "В новой форме будет отдельное приложение с графиком и ответственностью за просрочку.",
    "Следующий вопрос касается защиты персональных данных.",
    "Недавно изменились требования к уведомлению о начале обработки данных.",
    "Нам нужно проверить, все ли подразделения направили актуальные сведения.",
    "Ответственным за эту проверку предлагаю назначить специалиста по комплаенсу.",
    "Теперь коротко о бюджете департамента на следующий год.",
    "Основные расходы связаны с внешними консультантами и судебными издержками.",
    "Мы постараемся сократить расходы на консультантов за счёт собственной экспертизы.",
    "Для этого нужно развивать компетенции внутри команды и делиться опытом.",
    "Предлагаю раз в месяц проводить короткие внутренние семинары.",
    "Каждый юрист может рассказать о сложном деле, которое он недавно вёл.",
    "Это поможет всем быстрее разбираться в новых вопросах.",
    "Есть ли возражения или дополнения по этому предложению?",
    "Хорошо, тогда считаем, что решение принято.",
    "Последний вопрос касается графика отпусков на летний период.",
    "Прошу всех до конца месяца согласовать свои даты с руководителем группы.",
    "Важно, чтобы в каждой группе постоянно оставался хотя бы один сотрудник.",
    "На этом повестка исчерпана, спасибо всем за участие.",
]

# --- английская речь для чисто английских записей ---
EN_TALK = [
    "Thanks for joining the call, let's go through the agenda quickly.",
    "First, I would like to share an update on the contract review project.",
    "We started in March with a small team of three lawyers and one analyst.",
    "The main goal was to cut the time we spend on standard agreements.",
    "We mapped the whole process and found several steps that nobody really needed.",
    "Then we built a simple checklist and a library of approved clauses.",
    "The business can now prepare most drafts without waiting for legal.",
    "Of course, complex deals still come to us, and that is fine.",
    "The second topic is the training program for new hires.",
    "We recorded short videos about signatures, powers of attorney and archives.",
    "Feedback so far has been positive, but people want more practical examples.",
    "Let me know if you have a case that would work well for the next session.",
    "Finally, a quick note about the budget for next year.",
    "We expect to spend less on external counsel and a bit more on tools.",
    "I will send the detailed numbers by the end of the week.",
    "Does anyone have questions before we wrap up?",
]

# --- смесь в одном предложении: русская рамка + английское слово ---
MIX = [
    ("Давай повторим слово", "apple"),
    ("Теперь скажи по-английски слово", "window"),
    ("Как правильно произносится", "schedule"),
    ("Запиши в тетрадь слово", "deadline"),
    ("Это слово пишется так:", "receipt"),
    ("Повтори за мной, пожалуйста,", "comfortable"),
    ("Мы уже проходили глагол", "borrow"),
    ("Последнее слово на сегодня", "weather"),
]


def say(text, voice, rate=None):
    """Синтез фразы -> float32 16 кГц моно, тишина по краям обрезана."""
    with tempfile.TemporaryDirectory() as td:
        aiff = Path(td) / "x.aiff"
        cmd = ["say", "-v", voice, "-o", str(aiff)]
        if rate:
            cmd += ["-r", str(rate)]
        subprocess.run(cmd + [text], check=True)
        raw = subprocess.run(["ffmpeg", "-nostdin", "-i", str(aiff), "-ar", str(SR), "-ac", "1",
                              "-f", "f32le", "-loglevel", "error", "-"], capture_output=True, check=True).stdout
    a = np.frombuffer(raw, dtype=np.float32).copy()
    env = np.abs(a)
    thr = max(env.max() * 0.02, 1e-4)
    idx = np.where(env > thr)[0]
    if len(idx):
        a = a[max(0, idx[0] - int(0.03 * SR)): idx[-1] + int(0.05 * SR)]
    return a


class Track:
    """Собирает запись по фразам и ведёт эталон."""

    def __init__(self, rng, noise_db=-58.0):
        self.parts, self.t, self.phrases, self.rng, self.noise_db = [], 0.0, [], rng, noise_db

    def pause(self, sec):
        self.parts.append(np.zeros(int(sec * SR), dtype=np.float32))
        self.t += sec

    def add(self, audio, lang, text, **extra):
        self.parts.append(audio)
        dur = len(audio) / SR
        self.phrases.append(dict(start=round(self.t, 3), end=round(self.t + dur, 3), lang=lang, text=text, **extra))
        self.t += dur

    def save(self, name, gain_db=0.0, meta=None):
        a = np.concatenate(self.parts) if self.parts else np.zeros(SR, np.float32)
        a = a / max(1e-6, float(np.abs(a).max())) * 0.7
        a = a * (10 ** (gain_db / 20))
        # лёгкий фоновый шум - как у реального микрофона (без него детектору речи слишком просто)
        noise = self.rng.standard_normal(len(a)).astype(np.float32)
        noise = np.convolve(noise, np.ones(8) / 8, mode="same")
        noise *= (10 ** ((self.noise_db + gain_db) / 20)) / max(1e-9, float(np.sqrt(np.mean(noise ** 2))))
        a = np.clip(a + noise, -1, 1).astype(np.float32)
        OUT.mkdir(parents=True, exist_ok=True)
        wav = OUT / f"{name}.f32"
        wav.write_bytes(a.tobytes())
        subprocess.run(["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1",
                        "-i", str(wav), "-c:a", "aac", "-b:a", "96k", str(OUT / f"{name}.m4a")], check=True)
        wav.unlink()
        info = dict(name=name, duration=round(len(a) / SR, 3), phrases=self.phrases, **(meta or {}))
        (OUT / f"{name}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
        rms = 20 * np.log10(np.sqrt(np.mean(a ** 2)) + 1e-12)
        print(f"{name}.m4a  {len(a) / SR / 60:.1f} мин  фраз: {len(self.phrases)}  RMS {rms:.1f} dBFS")


def build_lesson(rng):
    tr = Track(rng)
    ru, en = LESSON_RU[:], LESSON_EN[:]
    rng.shuffle(en)
    lang = "ru"
    tr.pause(0.6)
    while ru or en:
        pool = ru if lang == "ru" else en
        if not pool:
            lang = "en" if lang == "ru" else "ru"
            continue
        for _ in range(int(rng.integers(1, 4))):
            if not pool:
                break
            text = pool.pop(0)
            voice = RU_VOICE if lang == "ru" else EN_VOICES[int(rng.integers(0, len(EN_VOICES)))]
            tr.add(say(text, voice, rate=int(rng.integers(170, 200))), lang, text)
            tr.pause(float(rng.uniform(0.2, 1.5)))
        lang = "en" if lang == "ru" else "ru"
    tr.save("lesson", meta=dict(note="ru/en через 1-3 предложения, паузы 0,2-1,5 с"))


def build_long_insert(rng, minutes=10.0, insert_at=0.5):
    tr = Track(rng)
    clips, total = [], 0.0
    i = 0
    while total < minutes * 60 - 45:
        text = LONG_RU[i % len(LONG_RU)]
        a = say(text, RU_VOICE, rate=int(rng.integers(165, 195)))
        p = float(rng.uniform(0.3, 1.2))
        clips.append((a, text, p))
        total += len(a) / SR + p
        i += 1
    cut = int(len(clips) * insert_at)
    tr.pause(0.5)
    for k, (a, text, p) in enumerate(clips):
        if k == cut:
            ins_start = tr.t
            for text_en in INSERT_EN:
                tr.add(say(text_en, "Samantha", rate=185), "en", text_en, insert=True)
                tr.pause(float(rng.uniform(0.3, 0.8)))
            ins_end = tr.t
        tr.add(a, "ru", text)
        tr.pause(p)
    tr.save("long_insert", meta=dict(insert=[round(ins_start, 2), round(ins_end, 2)]))


def build_mix(rng):
    tr = Track(rng)
    tr.pause(0.5)
    for ru_part, word in MIX:
        a = say(ru_part, RU_VOICE, rate=180)
        b = say(word, "Samantha", rate=170)
        gap = np.zeros(int(0.08 * SR), dtype=np.float32)
        tr.add(np.concatenate([a, gap, b]), "mix", f"{ru_part} {word}", word=word)
        tr.pause(float(rng.uniform(0.8, 1.5)))
    tr.save("mix")


def build_mix_same(rng):
    """Та же смесь, но целиком одним голосом (Milena читает и английское слово) -
    ближе к живому учителю, чем вклейка чужим голосом."""
    tr = Track(rng)
    tr.pause(0.5)
    for ru_part, word in MIX:
        text = f"{ru_part} {word}."
        tr.add(say(text, RU_VOICE, rate=175), "mix", text, word=word)
        tr.pause(float(rng.uniform(0.8, 1.5)))
    tr.save("mix_same")


def build_english(rng, voices=None, name="english"):
    """Чисто английская запись (для проверки «Авто» без ложных срабатываний)."""
    tr = Track(rng)
    tr.pause(0.5)
    for text in EN_TALK + LESSON_EN[:20] + INSERT_EN:
        v = voices or EN_VOICES
        tr.add(say(text, v[int(rng.integers(0, len(v)))], rate=int(rng.integers(170, 195))), "en", text)
        tr.pause(float(rng.uniform(0.3, 1.2)))
    tr.save(name)


def build_quiet(rng):
    tr = Track(rng, noise_db=-50.0)
    tr.pause(0.5)
    for text in LONG_RU[:24]:
        tr.add(say(text, RU_VOICE, rate=180), "ru", text)
        tr.pause(float(rng.uniform(0.4, 1.0)))
    # громкость около -35 dB (RMS речи), шум ниже на 15 dB
    tr.save("quiet", gain_db=-17.0, meta=dict(note="тихая запись, RMS около -35 dBFS"))


def build_dictation(rng):
    """~5 мин для диктофона: фразы с паузами, сплошная речь без пауз, короткое «Спасибо.»."""
    tr = Track(rng, noise_db=-50.0)
    tr.pause(1.0)
    for text in LONG_RU[:22]:                          # с паузами 0,8-1,6 с
        tr.add(say(text, RU_VOICE, rate=180), "ru", text, part="pauses")
        tr.pause(float(rng.uniform(0.8, 1.6)))
    tr.add(say("Спасибо.", RU_VOICE, rate=180), "ru", "Спасибо.", part="thanks")
    tr.pause(1.5)
    for text in LONG_RU[22:]:                           # почти без пауз: 0,1-0,3 с
        tr.add(say(text, RU_VOICE, rate=195), "ru", text, part="nopause")
        tr.pause(float(rng.uniform(0.1, 0.3)))
    tr.pause(1.0)
    for text in LONG_RU[:12]:                           # снова с паузами
        tr.add(say(text, RU_VOICE, rate=185), "ru", text, part="pauses2")
        tr.pause(float(rng.uniform(0.8, 1.4)))
    tr.save("dictation", meta=dict(note="для диктофона: паузы, без пауз, «Спасибо.»"))


def main():
    rng = np.random.default_rng(22)
    random.seed(22)
    which = set(sys.argv[1:]) or {"lesson", "long_insert", "mix", "mix_same", "quiet", "dictation",
                                  "english", "english_accent"}
    if "lesson" in which:
        build_lesson(rng)
    if "long_insert" in which:
        build_long_insert(rng)
    if "mix" in which:
        build_mix(rng)
    if "mix_same" in which:
        build_mix_same(np.random.default_rng(3))
    if "quiet" in which:
        build_quiet(rng)
    if "english" in which:
        build_english(np.random.default_rng(11))
    if "english_accent" in which:
        # английский, прочитанный русским голосом, - как английская речь с русским акцентом
        build_english(np.random.default_rng(12), voices=[RU_VOICE], name="english_accent")
    if "dictation" in which:
        build_dictation(np.random.default_rng(5))


if __name__ == "__main__":
    main()
