"""Hulpfuncties voor het notebook 'AI en erfgoed: wat kan er misgaan?'.

Studenten hoeven dit bestand niet te lezen: het notebook roept deze functies aan.
"""
import difflib
import html
import json
import re
from pathlib import Path

import pandas as pd
from IPython.display import HTML, Image, display

DATA = Path(__file__).parent / "data"

# Kleuren voor entiteiten
KLEUREN = {"PER": "#ffd6a5", "LOC": "#a0e7e5", "ORG": "#bdb2ff", "MISC": "#e4e4e4"}


# ---------------------------------------------------------------- algemeen
def lees_tekst(pad):
    """Leest een tekstbestand uit de datamap."""
    return (DATA / pad).read_text(encoding="utf-8")


def lees_json(pad):
    return json.loads((DATA / pad).read_text(encoding="utf-8"))


def toon_scan(pad, breedte=900, onderschrift=None):
    """Toont een scan uit de datamap, eventueel met een onderschrift."""
    display(Image(filename=str(DATA / pad), width=breedte))
    if onderschrift:
        display(HTML(f"<div style='font-size:12px;color:#666;font-style:italic;margin:2px 0 12px'>"
                     f"{html.escape(onderschrift)}</div>"))


def kader(tekst, titel=None, kleur="#f4f1ea", markeer=None):
    """Toont een stuk tekst in een leesbaar kader. `markeer` = woord dat geel gemarkeerd wordt."""
    kop = (f"<div style='font-family:system-ui,sans-serif;font-size:13px;font-weight:700;"
           f"margin-bottom:6px;color:#444'>{html.escape(titel)}</div>") if titel else ""
    inhoud = html.escape(tekst)
    if markeer:
        inhoud = inhoud.replace(html.escape(markeer),
                                f"<mark style='background:#ffe066;padding:0 3px;border-radius:3px;"
                                f"font-weight:700'>{html.escape(markeer)}</mark>", 1)
    inhoud = inhoud.replace("\n", "<br>")
    display(HTML(
        f"<div style='background:{kleur};color:#222;border-radius:8px;padding:12px 16px;margin:6px 0;"
        f"font-family:Georgia,serif;line-height:1.55;max-width:900px'>{kop}{inhoud}</div>"))


LABEL_KLEUREN = {"PER": "#ffd6a5", "PER_NOM": "#ffe8cc", "LOC": "#a0e7e5", "ORG": "#bdb2ff",
                 "ART": "#caffbf", "ART_NOM": "#e4ffdf", "—": "#f1f1f1"}


def toon_annotaties(verschillen):
    """Toont de labels van de vier annotatoren als gekleurde tabel."""
    df = pd.DataFrame(verschillen)[["brief", "woord", "Evelien", "Fien", "Gaetan", "Rafael"]]
    df.index = range(1, len(df) + 1)
    kleur = lambda v: f"background:{LABEL_KLEUREN.get(v, '#fff')};text-align:center;font-weight:600"
    display(df.style.map(kleur, subset=["Evelien", "Fien", "Gaetan", "Rafael"]))


# ---------------------------------------------------------------- deel 1: HTR
def _tokens(tekst):
    return re.findall(r"\S+|\n", tekst)


def vergelijk_transcripties(model, mens):
    """Zet de transcriptie van het model en die van de mens naast elkaar.

    Rood = staat alleen in de modeltranscriptie, groen = staat alleen in de menselijke.
    """
    a, b = _tokens(model), _tokens(mens)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    links, rechts = [], []

    def fmt(tokens, kleur=None):
        uit = []
        for t in tokens:
            if t == "\n":
                uit.append("<br>")
            elif kleur:
                uit.append(f"<span style='background:{kleur};border-radius:3px'>{html.escape(t)}</span>")
            else:
                uit.append(html.escape(t))
        return " ".join(uit)

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            links.append(fmt(a[i1:i2]))
            rechts.append(fmt(b[j1:j2]))
        else:
            links.append(fmt(a[i1:i2], "#ffb3b3"))
            rechts.append(fmt(b[j1:j2], "#b9f2b9"))
    stijl = ("vertical-align:top;padding:10px 14px;font-family:Georgia,serif;font-size:13px;"
             "line-height:1.45;background:#fbfaf7;color:#222;width:50%")
    display(HTML(
        "<table style='border-collapse:separate;border-spacing:8px;max-width:1100px'>"
        "<tr><th style='text-align:left'>🤖 Model (automatische transcriptie)</th>"
        "<th style='text-align:left'>🧑 Mens (gecorrigeerde transcriptie)</th></tr>"
        f"<tr><td style='{stijl}'>{' '.join(links)}</td><td style='{stijl}'>{' '.join(rechts)}</td></tr></table>"))


def overeenkomst(model, mens):
    """Percentage tekens dat overeenkomt tussen twee transcripties."""
    return round(100 * difflib.SequenceMatcher(None, model, mens, autojunk=False).ratio(), 1)


def laatste_regels(tekst, n):
    return "\n".join(tekst.strip().splitlines()[-n:])


# ---------------------------------------------------------------- modellen
_cache = {}


def _stil():
    """Verbergt technische meldingen en voortgangsbalken van de modelbibliotheek."""
    try:
        from transformers.utils import logging as tlog
        tlog.set_verbosity_error()
        tlog.disable_progress_bar()
    except Exception:
        pass
    try:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    except Exception:
        pass


def ner_model():
    """Laadt een meertalig model voor named entity recognition (personen, plaatsen, ...)."""
    if "ner" not in _cache:
        _stil()
        from transformers import pipeline
        _cache["ner"] = pipeline("token-classification", model="Babelscape/wikineural-multilingual-ner",
                                 aggregation_strategy="first")
    return _cache["ner"]


def sentiment_model():
    """Laadt een meertalig sentimentmodel dat 1 tot 5 sterren geeft."""
    if "sent" not in _cache:
        _stil()
        from transformers import pipeline
        _cache["sent"] = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")
    return _cache["sent"]


def invul_model():
    """Laadt een Frans taalmodel dat een ontbrekend woord invult."""
    if "mask" not in _cache:
        _stil()
        from transformers import pipeline
        _cache["mask"] = pipeline("fill-mask", model="camembert-base")
    return _cache["mask"]


# ---------------------------------------------------------------- NER tonen
def zoek_entiteiten(tekst):
    """Laat het NER-model personen (PER), plaatsen (LOC), organisaties (ORG) en overige (MISC) zoeken."""
    # Het model kan maar een beperkt stuk tekst tegelijk lezen: we geven het de tekst per zin.
    ents = []
    for m in re.finditer(r"[^.!?\n]+[.!?]*", tekst):
        zin = m.group(0)
        if not zin.strip():
            continue
        for e in ner_model()(zin):
            s, t = m.start() + e["start"], m.start() + e["end"]
            ents.append({"tekst": tekst[s:t], "label": e["entity_group"],
                         "zekerheid": round(float(e["score"]), 2), "start": s, "einde": t})
    return ents


def toon_entiteiten(tekst, ents=None):
    """Kleurt de gevonden entiteiten in de tekst."""
    if ents is None:
        ents = zoek_entiteiten(tekst)
    stukken, pos = [], 0
    for e in sorted(ents, key=lambda x: x["start"]):
        stukken.append(html.escape(tekst[pos:e["start"]]))
        kleur = KLEUREN.get(e["label"], "#eee")
        stukken.append(f"<span style='background:{kleur};padding:1px 3px;border-radius:4px'>"
                       f"{html.escape(tekst[e['start']:e['einde']])}"
                       f"<sub style='font-size:9px;font-weight:700'> {e['label']}</sub></span>")
        pos = e["einde"]
    stukken.append(html.escape(tekst[pos:]))
    legende = " ".join(f"<span style='background:{k};padding:1px 6px;border-radius:4px'>{l}</span>"
                       for l, k in KLEUREN.items())
    display(HTML(f"<div style='margin-bottom:6px;font-size:12px'>{legende}</div>"
                 f"<div style='background:#fbfaf7;color:#222;padding:12px 16px;border-radius:8px;"
                 f"font-family:Georgia,serif;line-height:1.7;max-width:900px'>"
                 f"{''.join(stukken).replace(chr(10), '<br>')}</div>"))
    return ents


# ---------------------------------------------------------------- deel 2: personenlijst
def personen():
    return pd.read_csv(DATA / "navez" / "personen_selectie.csv").fillna("")


# ---------------------------------------------------------------- deel 2: sentiment
STERREN = {"1 star": "1 ★ heel negatief", "2 stars": "2 ★★ negatief", "3 stars": "3 ★★★ neutraal",
           "4 stars": "4 ★★★★ positief", "5 stars": "5 ★★★★★ heel positief"}


def sentiment(tekst):
    """Geeft het oordeel van het sentimentmodel (1 = heel negatief, 5 = heel positief)."""
    r = sentiment_model()(tekst[:2000], truncation=True)[0]
    return STERREN.get(r["label"], r["label"])


def sentiment_tabel(zinnen):
    rijen = [{"tekst": z, "model zegt": sentiment(z)} for z in zinnen]
    df = pd.DataFrame(rijen)
    with pd.option_context("display.max_colwidth", None):
        display(df.style.hide(axis="index").set_properties(subset=["tekst"], **{"text-align": "left", "white-space": "normal"}))
    return df


def vul_in(zin, n=5):
    """Laat het taalmodel het woord op de plaats van <mask> raden."""
    return [f"{r['token_str'].strip()} ({r['score']:.0%})" for r in invul_model()(zin, top_k=n)]


# ---------------------------------------------------------------- deel 4: stochasticiteit (lokaal)
def trek_woorden(zin, keer=10, temperatuur=1.0, kandidaten=20):
    """Gooit 'de dobbelsteen': trekt telkens één woord volgens de kansen van het model.

    Zo kiest een generatief model (zoals ChatGPT) elk volgend woord.
    Temperatuur laag (bv. 0.1) = bijna altijd het meest waarschijnlijke woord.
    Temperatuur hoog (bv. 2) = veel meer toeval.
    """
    import numpy as np
    opties = invul_model()(zin, top_k=kandidaten)
    woorden = [o["token_str"].strip() for o in opties]
    p = np.array([o["score"] for o in opties], dtype=float)
    p = p ** (1.0 / max(temperatuur, 1e-3))
    p = p / p.sum()
    return [str(w) for w in np.random.choice(woorden, size=keer, p=p)]


# ---------------------------------------------------------------- generatief model (OpenRouter)
MODELLEN = ["openai/gpt-4o-mini", "anthropic/claude-haiku-5.5", "google/gemini-3.8-flash",
            "mistralai/mistral-small-3.2-24b-instruct", "meta-llama/llama-3.3-70b-instruct"]


def stel_sleutel_in():
    """Vraagt de OpenRouter-sleutel (die je van de lesgevers kreeg) zonder ze op het scherm te tonen."""
    import getpass
    import os
    try:
        sleutel = getpass.getpass("Plak hier de sleutel en druk op Enter: ").strip()
    except Exception:
        sleutel = ""
    if sleutel:
        os.environ["OPENROUTER_API_KEY"] = sleutel
        print("✅ Sleutel ingesteld.")
    else:
        print("Geen sleutel ingegeven: de cellen met een chatbot worden overgeslagen.")


SLEUTELBESTAND = Path(__file__).parent / "openrouter_key.txt"


def laad_sleutel_uit_bestand():
    """Laadt de sleutel uit 'openrouter_key.txt' (zit in de zip van de lesgevers), als die er is."""
    import os
    if not os.environ.get("OPENROUTER_API_KEY") and SLEUTELBESTAND.exists():
        sleutel = SLEUTELBESTAND.read_text(encoding="utf-8").strip()
        if sleutel:
            os.environ["OPENROUTER_API_KEY"] = sleutel
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def heeft_sleutel():
    import os
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def vraag(prompt, model="openai/gpt-4o-mini", temperatuur=1.0, systeem=None, max_tokens=400):
    """Stelt een vraag aan een generatief taalmodel via OpenRouter en geeft het antwoord terug."""
    import os
    import urllib.error
    import urllib.request
    if not heeft_sleutel():
        return "⚠️ Geen sleutel ingesteld. Voer eerst de cel met stel_sleutel_in() uit."
    berichten = ([{"role": "system", "content": systeem}] if systeem else []) + [{"role": "user", "content": prompt}]
    verzoek = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps({"model": model, "messages": berichten, "temperature": temperatuur,
        # minstens 800 tokens: sommige modellen "denken" eerst na voor ze antwoorden
                         "max_tokens": max(max_tokens, 800),
                         "reasoning": {"effort": "low"}}).encode("utf-8"),
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                 "Content-Type": "application/json"})
    try:
        try:
            with urllib.request.urlopen(verzoek, timeout=90) as antwoord:
                data = json.loads(antwoord.read().decode("utf-8"))
        except urllib.error.HTTPError as fout:
            data = json.loads(fout.read().decode("utf-8") or "{}") or {"error": {"message": str(fout)}}
        if "error" in data:
            return f"⚠️ Fout van OpenRouter: {data['error'].get('message', data['error'])}"
        inhoud = data["choices"][0]["message"].get("content")
        return inhoud.strip() if inhoud else "⚠️ Het model gaf een leeg antwoord. Probeer opnieuw of kies een ander model."
    except Exception as fout:  # netwerkproblemen e.d.
        return f"⚠️ Er ging iets mis: {fout}"


def toon_antwoord(prompt, model="openai/gpt-4o-mini", temperatuur=1.0, systeem=None):
    """Stelt de vraag en toont vraag en antwoord netjes."""
    kader(prompt, "❓ Vraag", kleur="#eef3fb")
    antwoord = vraag(prompt, model=model, temperatuur=temperatuur, systeem=systeem)
    kader(antwoord, f"🤖 {model} (temperatuur {temperatuur})", kleur="#f4f1ea")


def vraag_meermaals(prompt, keer=5, model="openai/gpt-4o-mini", temperatuur=1.0, systeem=None, max_tokens=400):
    """Stelt exact dezelfde vraag meerdere keren en toont alle antwoorden."""
    kader(prompt, "❓ Vraag (telkens dezelfde)", kleur="#eef3fb")
    antwoorden = []
    for i in range(keer):
        a = vraag(prompt, model=model, temperatuur=temperatuur, systeem=systeem, max_tokens=max_tokens)
        antwoorden.append(a)
        kader(a, f"Poging {i + 1}", kleur="#f4f1ea")
    return antwoorden


def vergelijk_modellen(prompt, modellen=None, temperatuur=1.0):
    """Stelt dezelfde vraag aan alle beschikbare modellen en toont de antwoorden onder elkaar."""
    kader(prompt, "❓ Vraag (aan elk model)", kleur="#eef3fb")
    for m in (modellen or MODELLEN):
        kader(vraag(prompt, model=m, temperatuur=temperatuur), f"🤖 {m}", kleur="#f4f1ea")


# ---------------------------------------------------------------- deel 2.3: counterfactuals
def sentiment_gemiddeld(tekst):
    """Gemiddelde score (1-5) volgens de kansen van het sentimentmodel: fijner dan het label alleen."""
    r = sentiment_model()(tekst[:2000], top_k=None, truncation=True)
    return round(sum(int(x["label"][0]) * x["score"] for x in r), 2)


def counterfactuals(paren):
    """Vergelijkt het sentiment van een zin met dezelfde zin waarin één woord vervangen is.

    paren: lijst van (zin, origineel woord, vervanging).
    """
    rijen = []
    for zin, oud, nieuw in paren:
        zin2 = zin.replace(oud, nieuw)
        a, b = sentiment_gemiddeld(zin), sentiment_gemiddeld(zin2)
        rijen.append({"zin": zin, "vervanging": f"{oud} → {nieuw}",
                      "label origineel": sentiment(zin), "label nieuw": sentiment(zin2),
                      "gemiddelde origineel": a, "gemiddelde nieuw": b, "verschil": round(b - a, 2)})
    df = pd.DataFrame(rijen)
    with pd.option_context("display.max_colwidth", None):
        display(df.style.hide(axis="index").format({"gemiddelde origineel": "{:.2f}", "gemiddelde nieuw": "{:.2f}", "verschil": "{:+.2f}"})
                .background_gradient(subset=["verschil"], cmap="RdYlGn", vmin=-1, vmax=1)
                .set_properties(subset=["zin"], **{"text-align": "left", "white-space": "normal"}))
    return df


def counterfactuals_chatbot(paren, model="openai/gpt-4o-mini"):
    """Laat een chatbot beide versies van elke zin scoren (1-5)."""
    v = "Geef deze zin een sentimentscore van 1 (heel negatief) tot 5 (heel positief). Antwoord met enkel een cijfer.\n\n"
    rijen = []
    for zin, oud, nieuw in paren:
        zin2 = zin.replace(oud, nieuw)
        rijen.append({"zin": zin, "vervanging": f"{oud} → {nieuw}",
                      "chatbot origineel": vraag(v + zin, model=model, temperatuur=0, max_tokens=5),
                      "chatbot nieuw": vraag(v + zin2, model=model, temperatuur=0, max_tokens=5)})
    df = pd.DataFrame(rijen)
    with pd.option_context("display.max_colwidth", None):
        display(df.style.hide(axis="index").set_properties(subset=["zin"], **{"text-align": "left", "white-space": "normal"}))
    return df
