"""Chunk-seam reconciliation for the chunked-transcription join.

When the VAD cuts mid-sentence (an intra-thought pause), the two chunks
straddling the cut are transcribed somewhat independently. Whisper's
decisions at the seam can disagree with the fact that it is ONE
continuous sentence, producing three defect classes that the naive
`" ".join` leaves in the final text:

  TERM — chunk N ends with a sentence terminator (.!?) it invented from
         the acoustic fade, while chunk N+1 — seeing N's tail as rolling
         prompt — correctly starts lowercase. The terminator is spurious.
         Fix: drop it (N+1 has more context, trust the continuation).

  DUP  — chunk N's last word reappears as chunk N+1's first word (the
         token is in the rolling prompt AND re-emitted at the next
         onset): "...происходит Происходит транскрибация".
         Fix: drop the duplicated leading word of N+1.

  CAP  — chunk N did NOT terminate, yet chunk N+1 starts Capitalized,
         rendering a mid-sentence pause as a false sentence break — the
         "рваность" users report: "...чтобы Супруга...".
         Fix: lowercase N+1's first letter, but ONLY when the word is a
         common Russian non-name word (in the _COMMON_RU whitelist). A
         name/place/term ("Елена", "Стоктон", "Масло", "Python") is left
         capitalized — we never lowercase an unknown word, because a
         seam can legitimately fall right before a proper noun.

This module is dependency-light on purpose (only `re` + the vocabulary
reader) so both pipeline.finalize() and the offline replay/validation
tools can import and apply the exact same reconciliation.
"""

import re
from typing import List, Optional, Set

# A "word" — a run of Unicode letters, optionally hyphenated (so
# "mlx-whisper" / "по-моему" count as one word). Excludes digits and
# underscores so "v15" or stray markers don't masquerade as words.
_WORD = r"[^\W\d_]+(?:-[^\W\d_]+)*"
_FIRST_WORD_RE = re.compile(r"[\W\d_]*(" + _WORD + r")")
_WORD_RE = re.compile(_WORD)
_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
_TERMINATORS = ".!?"
# Separators left dangling after a removed leading duplicate word.
_DANGLING = " \t,.;:!?—–-"

# Closed whitelist of common Russian words that are NEVER proper nouns —
# prepositions, conjunctions, pronouns/determiners (with their frequent
# inflections), particles, discourse markers, quantifiers and auxiliary
# verbs. The CAP fix lowercases a capitalized seam-initial word ONLY if
# its lowercase form is in this set. This is a WHITELIST by design: an
# unknown word (a name like "Елена", a place like "Стоктон", a term like
# "Масло") is left capitalized rather than risk lowercasing a proper
# noun. The cost is that some genuine common words missing from the list
# keep a spurious capital — a minor cosmetic miss, far safer than
# mangling a name. Grow the list from telemetry as misses surface.
_COMMON_RU = frozenset("""
и а но или либо да что чтобы как когда если пока потому поэтому хотя тоже
также ведь зато причем причём итак значит следовательно иначе словно будто
точно чем нежели раз коли дабы тем
в во на над под с со к ко по при для от ото до из изо за о об обо про без
безо через сквозь между меж перед передо около возле после у вокруг кроме
вместо насчет насчёт благодаря согласно среди средь вдоль мимо ради против
напротив внутри снаружи поверх вне
я ты он она оно они мы вы меня тебя его её ее нас вас их мне тебе ему ей нам
вам им мной тобой нами вами ними нем нём ней них него нее неё нему
это этот эта эти этого этой этих этом этим эту тот та те то того той тех том
тем ту такой такая такое такие такого таких таком таким
который которая которое которые которого которой которых котором которым
кто чей чья чьё чьи весь вся всё все всего всей всех всем всеми
сам сама само сами самого самой самих себя себе собой свой своя своё свои
своего своей своих мой моя моё мои моего твой твоя твоё наш наша наше наши
нашего ваш ваша ваше ваши каждый каждая каждое каждые каждого любой любая
любое любые всякий иной иная другой другая другое другие некоторый некоторые
никто ничто ничего никого что-то кто-то что-нибудь кто-нибудь чего кого кому
чему нечто некто
ну вот же ли бы не ни разве неужели именно даже лишь только уже ещё еще
вообще например скажем видимо кажется конечно наверное наверно возможно
допустим короче типа кстати впрочем действительно собственно как-то
как-нибудь так там тут здесь тогда теперь сейчас потом затем сначала опять
снова вернее точнее почему зачем почти совсем очень более менее можно нужно
надо нельзя давай давайте пусть пускай наконец ладно хорошо окей
несколько много мало немного столько сколько больше меньше чуть примерно
приблизительно
есть был была было были будет будут будем буду будешь будете быть нет
""".split())

# Minimum word length for the DUP check — short function words ("и",
# "а", "в", "по") legitimately repeat across a pause ("...по по-моему")
# far more often than they are seam artifacts, so only treat words of
# 3+ letters as duplicates.
_MIN_DUP_LEN = 3


def _first_word(s: str) -> Optional[str]:
    m = _FIRST_WORD_RE.match(s)
    return m.group(1) if m else None


def _last_word(s: str) -> Optional[str]:
    words = _WORD_RE.findall(s)
    return words[-1] if words else None


def _strip_leading_word(s: str) -> str:
    """Remove the first word (and leading junk + trailing separators) from s."""
    m = _FIRST_WORD_RE.match(s)
    if not m:
        return s
    return s[m.end():].lstrip(_DANGLING)


def _lower_first_letter(s: str) -> str:
    for i, ch in enumerate(s):
        if ch.isalpha():
            return s[:i] + ch.lower() + s[i + 1:]
    return s


def _is_false_capital(word: str, vocab: Set[str]) -> bool:
    """True if `word` is a capitalized seam-initial word that is safe to
    lowercase as a sentence continuation. WHITELIST semantics — only common
    Russian non-name words qualify:
      - title-case only (first upper, rest lower or single letter) — never
        touch ALL-CAPS acronyms ("УЗИ");
      - lowercase form is in the closed common-word set (_COMMON_RU) — a
        name/place/term is NOT in it, so it stays capitalized;
      - never lowercase a word the user put in their vocabulary.
    """
    if not word or not word[0].isupper():
        return False
    if len(word) > 1 and not word[1:].islower():
        return False
    lw = word.lower()
    if lw in vocab:
        return False
    return lw in _COMMON_RU


def reconcile_seams(ordered: List[str], vocab: Optional[Set[str]] = None) -> List[str]:
    """Reconcile chunk-boundary inconsistencies in a list of per-chunk
    texts. Returns a NEW list (does not mutate the input). Apply before
    joining with single spaces.
    """
    vocab = vocab or set()
    out = list(ordered)
    for i in range(len(out) - 1):
        prev, curr = out[i], out[i + 1]
        if not prev or not curr:
            continue
        prev_s, curr_s = prev.rstrip(), curr.lstrip()
        if not prev_s or not curr_s:
            continue

        prev_last = _last_word(prev_s)
        curr_first = _first_word(curr_s)
        if not prev_last or not curr_first:
            continue
        prev_terminated = prev_s[-1] in _TERMINATORS

        # DUP — drop the duplicated leading word of the next chunk.
        if len(prev_last) >= _MIN_DUP_LEN and prev_last.lower() == curr_first.lower():
            curr_s = _strip_leading_word(curr_s)
            if not curr_s:
                out[i + 1] = curr_s
                continue
            curr_first = _first_word(curr_s)
            if not curr_first:
                out[i + 1] = curr_s
                continue

        # TERM / CAP — reconcile the casing across the seam.
        if prev_terminated and curr_s[0].islower():
            prev_s = prev_s[:-1].rstrip()
        elif not prev_terminated and _is_false_capital(curr_first, vocab):
            curr_s = _lower_first_letter(curr_s)

        out[i] = prev_s
        out[i + 1] = curr_s
    return out


def vocabulary_word_set() -> Set[str]:
    """Lowercased set of individual words across all vocabulary entries
    (multi-word phrases are split). Used to protect proper nouns from the
    CAP lowercasing. Never raises — empty set on any error."""
    try:
        from .vocabulary import read_vocabulary
        words: Set[str] = set()
        for entry in read_vocabulary():
            for w in _WORD_RE.findall(entry.lower()):
                words.add(w)
        return words
    except Exception:
        return set()
