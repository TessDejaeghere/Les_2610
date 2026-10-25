# AI en erfgoed: hands-on

Materiaal voor de hands-on sessie van de les *Erfgoed en Digitale Transformatie* (Universiteit Gent, oktober 2026), door Tess Dejaeghere en Bas Vercruysse.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TessDejaeghere/Les_2610/blob/main/AI_en_erfgoed_hands-on.ipynb)

## Starten

1. Klik op **Open in Colab** hierboven.
2. Voer **Stap 1** uit: het notebook haalt de teksten, scans en hulpfuncties automatisch op uit deze repo.
3. Voer **Stap 1b** uit en kies `les_extra.zip` (krijg je van de lesgevers). Daarin zitten de sleutel voor de chatbot en één fragment dat niet publiek mag. Zonder dat bestand werkt de rest van het notebook gewoon.

Of zelf binnenhalen:

```bash
wget https://github.com/TessDejaeghere/Les_2610/archive/refs/heads/main.zip
```

## Inhoud

| Map | Wat |
|---|---|
| `AI_en_erfgoed_hands-on.ipynb` | Het notebook (Nederlandstalig, ongeveer 80 minuten) |
| `hulpfuncties.py` | Functies die het notebook gebruikt |
| `data/htr/` | Scans en transcripties van twee brieven aan F.-J. Navez (KBR, Ms. II 70) |
| `data/navez/` | Brieven en een personenselectie uit het Navez-corpus |
| `data/reisteksten/` | Fragmenten uit historische reisteksten (publiek domein) |
| `data/annotaties/` | Annotaties van vier annotatoren (NER, Navez-project) |

## Let op

- De teksten bevatten **racistische en seksistische historische taal**. Ze worden gebruikt om bias in AI-modellen te tonen.
- De scans komen van de **KBR** (Koninklijke Bibliotheek van België) en worden hier enkel voor onderwijs gebruikt.
- Er staan **geen API-sleutels** in deze repo.
