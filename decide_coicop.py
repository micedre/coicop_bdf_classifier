#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "openai>=1.55",
#   "pandas>=2.0",
#   "pyarrow>=14.0",
#   "pydantic>=2.0",
#   "tqdm>=4.0",
# ]
# ///
"""
Décision finale du code COICOP par LLM.

Usage — observation unique :
    uv run decide_coicop.py --id <observation_id>
    uv run decide_coicop.py --id <id> --model gpt-4o-mini --output json

Usage — traitement complet (toutes les observations → parquet) :
    uv run decide_coicop.py
    uv run decide_coicop.py --model gpt-4o-mini --concurrency 20
    uv run decide_coicop.py --output-file results/llm_decisions.parquet

    Le traitement complet supporte la reprise automatique : si le fichier de
    sortie existe déjà, les observations déjà traitées sont ignorées.

Variables d'environnement :
    OPENAI_API_KEY   (obligatoire)
    OPENAI_BASE_URL  (optionnel, pour proxy/endpoint custom)
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import openai
import pandas as pd
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

# ── Logger ────────────────────────────────────────────────────────────────────

logger = logging.getLogger("coicop_llm")


def setup_logging(console_level: str, log_file: Path | None) -> None:
    """Configure console + optional file logging.

    Console shows INFO+ by default (or DEBUG with --log-level debug).
    File always captures DEBUG (full request/response details).
    """
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.debug("Logging vers fichier : %s", log_file)

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent

CONFIDENCE_LABELS = {
    1: "Très faible — le code proposé est incertain",
    2: "Faible — doute significatif sur le classement",
    3: "Moyen — classement vraisemblable mais non certain",
    4: "Élevé — le code est probablement correct",
    5: "Très élevé — le code est certain",
}

CHECKPOINT_EVERY = 50  # sauvegardes intermédiaires toutes les N observations

# ── Schéma de réponse structurée ──────────────────────────────────────────────

class DecisionCoicop(BaseModel):
    coicop_code: str = Field(
        description=(
            "Code COICOP retenu, exactement tel qu'il figure dans la nomenclature "
            "fournie (ex. '01.1.2.3'). Doit être un code existant."
        )
    )
    libelle: str = Field(
        description="Libellé COICOP correspondant au code retenu, copié de la nomenclature."
    )
    explication: str = Field(
        description=(
            "Justification du choix en français : pourquoi ce code est retenu, "
            "comment les prédictions des modèles ont été prises en compte, "
            "et les éventuelles ambiguïtés."
        )
    )
    confiance: int = Field(
        ge=1, le=5,
        description=(
            "Score de confiance de 1 (très faible) à 5 (très élevé) "
            "sur la décision prise."
        )
    )

# ── Chargement des données ────────────────────────────────────────────────────

def load_nomenclature() -> pd.DataFrame:
    """Charge la nomenclature COICOP sous forme de DataFrame (Code, Libelle)."""
    csv_path = DATA_DIR / "coicop_et_codes_techniques.csv"
    return pd.read_csv(csv_path, sep=";", encoding="utf-8")


def _nomen_to_str(nomen: pd.DataFrame) -> str:
    return "\n".join(f"{r['Code']} — {r['Libelle']}" for _, r in nomen.iterrows())


def filter_nomenclature(nomen: pd.DataFrame, obs: dict, full: bool = False) -> str:
    """Retourne la nomenclature filtrée aux sections pertinentes pour cette observation.

    Strategy : collecte les préfixes L2 (ex. "01.1") de toutes les prédictions
    disponibles, puis garde :
      - les codes de section L1 pour le contexte
      - tous les codes dont le Code commence par l'un de ces préfixes L2

    Si aucune prédiction ou full=True, renvoie la nomenclature complète.
    Réduit typiquement ~700 lignes à ~50-150 lignes (gain ×4-10 sur les tokens).
    """
    if full:
        logger.debug("Nomenclature complète (%d codes)", len(nomen))
        return _nomen_to_str(nomen)

    predicted_codes = [
        obs.get("lcs_code"),
        obs.get("rag_code"),
        obs.get("ttc_code_1"),
        obs.get("ttc_code_2"),
        obs.get("ttc_code_3"),
    ]
    predicted_codes = [
        str(c) for c in predicted_codes
        if c is not None and not (isinstance(c, float) and pd.isna(c))
    ]

    if not predicted_codes:
        logger.debug("[%s] Aucune prédiction — nomenclature complète", obs.get("id"))
        return _nomen_to_str(nomen)

    # L1 prefixes (ex. "01") and L2 prefixes (ex. "01.1")
    l1 = {c.split(".")[0] for c in predicted_codes}
    l2 = {".".join(c.split(".")[:2]) for c in predicted_codes if "." in c}

    def is_relevant(code: str) -> bool:
        parts = code.split(".")
        if len(parts) == 1:                      # section header (ex. "01")
            return parts[0] in l1
        if len(parts) == 2:                      # sub-section (ex. "01.1")
            return parts[0] in l1
        return any(code.startswith(p) for p in l2)  # deeper codes

    mask = nomen["Code"].apply(is_relevant)
    filtered = nomen[mask]

    logger.debug(
        "[%s] Nomenclature filtrée : %d/%d codes (L1=%s, L2=%s)",
        obs.get("id"), len(filtered), len(nomen), sorted(l1), sorted(l2),
    )
    return _nomen_to_str(filtered)


def load_all_observations() -> pd.DataFrame:
    """Fusionne les trois fichiers parquet et retourne le DataFrame complet."""
    lcs_raw = pd.read_parquet(DATA_DIR / "predictions_lcs.parquet")
    rag_raw = pd.read_parquet(DATA_DIR / "predictions_rag.parquet")
    ttc_raw = pd.read_parquet(DATA_DIR / "predictions_ttc.parquet")

    lcs = lcs_raw[[
        "id", "raw_product", "l_pr_product", "shop", "shop_type_name", "budget",
        "code", "predict_code", "distance", "common_substring", "prop_in_s2",
    ]].rename(columns={
        "predict_code": "lcs_code",
        "distance":     "lcs_distance",
        "common_substring": "lcs_substring",
        "prop_in_s2":   "lcs_prop",
    })

    rag = rag_raw[["id", "code_predict", "confidence", "codable"]].rename(columns={
        "code_predict": "rag_code",
        "confidence":   "rag_confidence",
    })

    ttc_cols = ["id", "predicted_code", "confidence"]
    for i in range(2, 4):
        ttc_cols += [f"predicted_code_top{i}", f"confidence_top{i}"]
    ttc = ttc_raw[ttc_cols].rename(columns={
        "predicted_code":       "ttc_code_1",
        "confidence":           "ttc_conf_1",
        "predicted_code_top2":  "ttc_code_2",
        "confidence_top2":      "ttc_conf_2",
        "predicted_code_top3":  "ttc_code_3",
        "confidence_top3":      "ttc_conf_3",
    })

    return lcs.merge(rag, on="id", how="left").merge(ttc, on="id", how="left")


def get_observation(df: pd.DataFrame, obs_id: str) -> dict:
    row = df[df["id"] == obs_id]
    if row.empty:
        print(f"Erreur : identifiant '{obs_id}' introuvable.", file=sys.stderr)
        sys.exit(1)
    return row.iloc[0].to_dict()

# ── Construction du prompt ────────────────────────────────────────────────────

def build_prompt(obs: dict, nomenclature: pd.DataFrame, full_nomen: bool = False) -> str:
    def fmt(val, fmt_spec=None):
        if val is None:
            return "N/A"
        try:
            if pd.isna(val):
                return "N/A"
        except (TypeError, ValueError):
            pass
        if fmt_spec:
            return format(val, fmt_spec)
        return str(val)

    lcs_block = (
        f"  Code prédit  : {fmt(obs.get('lcs_code'))}\n"
        f"  Sous-chaîne  : {fmt(obs.get('lcs_substring'))} "
        f"(proportion dans référence : {fmt(obs.get('lcs_prop'), '.0%')})\n"
        f"  Distance     : {fmt(obs.get('lcs_distance'), '.3f')}"
    )

    rag_conf = obs.get("rag_confidence")
    rag_conf_str = f"{rag_conf:.0%}" if rag_conf is not None and not pd.isna(rag_conf) else "N/A"
    rag_block = (
        f"  Code prédit  : {fmt(obs.get('rag_code'))}\n"
        f"  Confiance    : {rag_conf_str}\n"
        f"  Codable      : {fmt(obs.get('codable'))}"
    )

    ttc_lines = []
    for rank in range(1, 4):
        code = fmt(obs.get(f"ttc_code_{rank}"))
        conf = obs.get(f"ttc_conf_{rank}")
        conf_str = f"{conf:.0%}" if conf is not None and not pd.isna(conf) else "N/A"
        ttc_lines.append(f"  Top {rank} : {code} (confiance {conf_str})")
    ttc_block = "\n".join(ttc_lines)

    return f"""\
Tu es un expert en comptabilité nationale et en statistiques de consommation.
Ta mission est de déterminer le code COICOP le plus approprié pour un produit acheté,
en t'appuyant sur le contexte d'achat et les prédictions de trois modèles automatiques.

═══════════════════════════════════════
PRODUIT
═══════════════════════════════════════
Libellé brut       : {fmt(obs.get('raw_product'))}
Libellé normalisé  : {fmt(obs.get('l_pr_product'))}
Enseigne           : {fmt(obs.get('shop'))}
Type de magasin    : {fmt(obs.get('shop_type_name'))}
Montant (€)        : {fmt(obs.get('budget'), '.2f')}
Code de référence  : {fmt(obs.get('code'))}

═══════════════════════════════════════
PRÉDICTIONS DES MODÈLES
═══════════════════════════════════════

[LCS — Longest Common Substring]
{lcs_block}

[RAG — Retrieval-Augmented Generation]
{rag_block}

[TTC — Classificateur par apprentissage profond]
{ttc_block}

═══════════════════════════════════════
NOMENCLATURE COICOP (codes valides)
═══════════════════════════════════════
{filter_nomenclature(nomenclature, obs, full=full_nomen)}

═══════════════════════════════════════
INSTRUCTIONS
═══════════════════════════════════════
1. Choisis UN code COICOP parmi ceux listés dans la nomenclature ci-dessus.
2. Le code doit correspondre au niveau le plus précis qui te semble justifié.
3. Explique en français comment tu as pesé les prédictions des trois modèles
   et tout autre indice (enseigne, type de magasin, montant).
4. Attribue un score de confiance de 1 (très faible) à 5 (très élevé).
"""

# ── Appel LLM (synchrone — mode observation unique) ──────────────────────────

SYSTEM_MSG = (
    "Tu es un expert COICOP. Réponds uniquement via le schéma JSON demandé. "
    "Le champ coicop_code doit être un code existant dans la nomenclature fournie."
)


def call_llm_sync(prompt: str, model: str, client: OpenAI) -> DecisionCoicop:
    logger.debug("Request: model=%s, prompt_chars=%d (~%d tokens)",
                 model, len(prompt), len(prompt) // 4)
    t0 = time.perf_counter()
    try:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user",   "content": prompt},
            ],
            response_format=DecisionCoicop,
            temperature=0.0,
        )
    except openai.AuthenticationError as exc:
        logger.error("Authentification échouée (vérifiez OPENAI_API_KEY) : %s", exc)
        raise
    except openai.RateLimitError as exc:
        logger.warning("Rate limit après %.2fs : %s", time.perf_counter() - t0, exc)
        raise
    except openai.APIStatusError as exc:
        logger.error("APIStatusError status=%d après %.2fs : %s",
                     exc.status_code, time.perf_counter() - t0, exc.message)
        raise
    except openai.APIConnectionError as exc:
        logger.error("Connexion impossible après %.2fs : %s", time.perf_counter() - t0, exc)
        raise

    latency = time.perf_counter() - t0
    usage = completion.usage
    logger.info(
        "Response: finish_reason=%s | tokens prompt=%d completion=%d total=%d | %.2fs",
        completion.choices[0].finish_reason,
        usage.prompt_tokens if usage else -1,
        usage.completion_tokens if usage else -1,
        usage.total_tokens if usage else -1,
        latency,
    )

    result = completion.choices[0].message.parsed
    if result is None:
        raw = completion.choices[0].message.content
        logger.error("Structured output parsing échoué. Réponse brute : %s", raw)
        raise ValueError(f"Structured output parsing failed. Raw: {raw}")

    logger.debug("Parsed: coicop_code=%s confiance=%d", result.coicop_code, result.confiance)
    return result

# ── Appel LLM (asynchrone — mode batch) ──────────────────────────────────────

async def call_llm_async(
    obs: dict,
    nomenclature: pd.DataFrame,
    model: str,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    full_nomen: bool = False,
) -> dict:
    """Traite une observation et retourne un dict résultat (erreur capturée)."""
    async with semaphore:
        obs_id = obs["id"]
        base = {
            "id":            obs_id,
            "llm_model":     model,
            "llm_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        prompt = build_prompt(obs, nomenclature, full_nomen=full_nomen)
        logger.debug("[%s] Request: model=%s, prompt_chars=%d (~%d tokens)",
                     obs_id, model, len(prompt), len(prompt) // 4)
        t0 = time.perf_counter()
        try:
            completion = await client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user",   "content": prompt},
                ],
                response_format=DecisionCoicop,
                temperature=0.0,
            )
            latency = time.perf_counter() - t0
            usage = completion.usage
            finish = completion.choices[0].finish_reason

            logger.info(
                "[%s] ✓ finish=%s | tokens prompt=%d completion=%d total=%d | %.2fs",
                obs_id, finish,
                usage.prompt_tokens if usage else -1,
                usage.completion_tokens if usage else -1,
                usage.total_tokens if usage else -1,
                latency,
            )

            decision = completion.choices[0].message.parsed
            if decision is None:
                raw = completion.choices[0].message.content
                logger.error("[%s] Structured output parsing échoué. Réponse brute : %s",
                             obs_id, raw)
                raise ValueError(f"Structured output parsing returned None. Raw: {raw}")

            logger.debug("[%s] Parsed: coicop_code=%s confiance=%d",
                         obs_id, decision.coicop_code, decision.confiance)

            return {
                **base,
                "llm_code":        decision.coicop_code,
                "llm_libelle":     decision.libelle,
                "llm_explication": decision.explication,
                "llm_confiance":   decision.confiance,
                "llm_prompt_tokens":     usage.prompt_tokens if usage else None,
                "llm_completion_tokens": usage.completion_tokens if usage else None,
                "llm_latency_s":         round(latency, 3),
                "llm_error":       None,
            }

        except openai.RateLimitError as exc:
            latency = time.perf_counter() - t0
            logger.warning("[%s] RateLimitError après %.2fs : %s", obs_id, latency, exc)
            return {**base, "llm_code": None, "llm_libelle": None,
                    "llm_explication": None, "llm_confiance": None,
                    "llm_prompt_tokens": None, "llm_completion_tokens": None,
                    "llm_latency_s": round(latency, 3),
                    "llm_error": f"RateLimitError: {exc}"}

        except openai.APIStatusError as exc:
            latency = time.perf_counter() - t0
            logger.error("[%s] APIStatusError status=%d après %.2fs : %s",
                         obs_id, exc.status_code, latency, exc.message)
            return {**base, "llm_code": None, "llm_libelle": None,
                    "llm_explication": None, "llm_confiance": None,
                    "llm_prompt_tokens": None, "llm_completion_tokens": None,
                    "llm_latency_s": round(latency, 3),
                    "llm_error": f"APIStatusError({exc.status_code}): {exc.message}"}

        except openai.APIConnectionError as exc:
            latency = time.perf_counter() - t0
            logger.error("[%s] APIConnectionError après %.2fs : %s", obs_id, latency, exc)
            return {**base, "llm_code": None, "llm_libelle": None,
                    "llm_explication": None, "llm_confiance": None,
                    "llm_prompt_tokens": None, "llm_completion_tokens": None,
                    "llm_latency_s": round(latency, 3),
                    "llm_error": f"APIConnectionError: {exc}"}

        except Exception as exc:
            latency = time.perf_counter() - t0
            logger.error("[%s] Erreur inattendue après %.2fs : %s: %s",
                         obs_id, latency, type(exc).__name__, exc, exc_info=True)
            return {**base, "llm_code": None, "llm_libelle": None,
                    "llm_explication": None, "llm_confiance": None,
                    "llm_prompt_tokens": None, "llm_completion_tokens": None,
                    "llm_latency_s": round(latency, 3),
                    "llm_error": f"{type(exc).__name__}: {exc}"}

# ── Mode batch ────────────────────────────────────────────────────────────────

def save_parquet(records: list[dict], path: Path) -> None:
    pd.DataFrame(records).to_parquet(path, index=False)


async def run_batch(
    df: pd.DataFrame,
    nomenclature: pd.DataFrame,
    model: str,
    concurrency: int,
    output_file: Path,
    full_nomen: bool = False,
) -> None:
    # ── Reprise sur checkpoint ──
    done_ids: set[str] = set()
    results: list[dict] = []

    if output_file.exists():
        existing = pd.read_parquet(output_file)
        done_ids = set(existing["id"].tolist())
        results = existing.to_dict(orient="records")
        logger.info("Reprise : %d observations déjà traitées, %d restantes.",
                    len(done_ids), len(df) - len(done_ids))

    pending = [
        row for row in df.to_dict(orient="records")
        if row["id"] not in done_ids
    ]

    if not pending:
        logger.info("Toutes les observations ont déjà été traitées.")
        return

    logger.info("Démarrage batch : %d observations, concurrence=%d, modèle=%s",
                len(pending), concurrency, model)

    # ── Client async ──
    api_key = os.environ["OPENAI_API_KEY"]
    kwargs: dict = {"api_key": api_key}
    if base_url := os.environ.get("OPENAI_BASE_URL"):
        kwargs["base_url"] = base_url
        logger.debug("OPENAI_BASE_URL=%s", base_url)
    client = AsyncOpenAI(**kwargs)

    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    n_errors = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    batch_t0 = time.perf_counter()

    output_file.parent.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(total=len(pending), desc="Traitement LLM", unit="obs", file=sys.stderr)

    async def process_and_save(obs: dict) -> None:
        nonlocal n_errors, total_prompt_tokens, total_completion_tokens
        result = await call_llm_async(obs, nomenclature, model, client, semaphore, full_nomen)
        async with lock:
            results.append(result)
            if result["llm_error"]:
                n_errors += 1
            else:
                total_prompt_tokens     += result.get("llm_prompt_tokens") or 0
                total_completion_tokens += result.get("llm_completion_tokens") or 0
            if len(results) % CHECKPOINT_EVERY == 0:
                save_parquet(results, output_file)
                elapsed = time.perf_counter() - batch_t0
                rate = len(results) / elapsed if elapsed > 0 else 0
                logger.info(
                    "Checkpoint %d/%d — %.1f obs/min — tokens cumulés prompt=%d completion=%d — %d erreurs",
                    len(results), len(pending) + len(done_ids),
                    rate * 60,
                    total_prompt_tokens, total_completion_tokens,
                    n_errors,
                )
                pbar.set_postfix(
                    tokens=total_prompt_tokens + total_completion_tokens,
                    errors=n_errors,
                )
        pbar.update(1)

    tasks = [process_and_save(obs) for obs in pending]
    await asyncio.gather(*tasks)
    pbar.close()

    save_parquet(results, output_file)

    elapsed = time.perf_counter() - batch_t0
    n_ok = len(results) - n_errors
    total_tokens = total_prompt_tokens + total_completion_tokens
    logger.info(
        "Batch terminé : %d succès, %d erreurs | tokens prompt=%d completion=%d total=%d | "
        "durée=%.1fs (%.1f obs/min) → %s",
        n_ok, n_errors,
        total_prompt_tokens, total_completion_tokens, total_tokens,
        elapsed, len(results) / elapsed * 60 if elapsed > 0 else 0,
        output_file,
    )

# ── Affichage (mode observation unique) ───────────────────────────────────────

def print_result(obs: dict, decision: DecisionCoicop) -> None:
    ref = obs.get("code")
    ref_str = str(ref) if ref is not None and not pd.isna(ref) else "N/A"
    correct = (
        str(decision.coicop_code).startswith(ref_str[:len(decision.coicop_code)])
        if ref_str != "N/A" else None
    )
    match_str = ""
    if correct is not None:
        match_str = "  ✓ correspond au code de référence" if correct else "  ✗ diffère du code de référence"

    conf_label = CONFIDENCE_LABELS.get(decision.confiance, "")

    print()
    print("══════════════════════════════════════════════════")
    print("  DÉCISION LLM")
    print("══════════════════════════════════════════════════")
    print(f"  Produit       : {obs.get('raw_product', 'N/A')}")
    print(f"  Enseigne      : {obs.get('shop', 'N/A')}")
    print()
    print(f"  Code COICOP   : {decision.coicop_code}")
    print(f"  Libellé       : {decision.libelle}")
    print(f"  Référence     : {ref_str}{match_str}")
    print()
    print(f"  Confiance     : {decision.confiance}/5 — {conf_label}")
    print()
    print("  Explication :")
    for line in decision.explication.split(". "):
        line = line.strip()
        if line:
            print(f"    {line}.")
    print("══════════════════════════════════════════════════")
    print()

# ── Point d'entrée ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Décision COICOP par LLM. Sans --id, traite toutes les observations "
            "et sauvegarde dans un fichier parquet (avec reprise automatique)."
        )
    )
    parser.add_argument(
        "--id", default=None,
        help="Identifiant d'une observation précise. Sans cet argument : mode batch.",
    )
    parser.add_argument(
        "--model", default="gpt-4o",
        help="Modèle OpenAI (défaut : gpt-4o).",
    )
    parser.add_argument(
        "--output", choices=["text", "json"], default="text",
        help="Format de sortie en mode observation unique (défaut : text).",
    )
    parser.add_argument(
        "--output-file", default=str(DATA_DIR / "predictions_llm.parquet"),
        help="Fichier parquet de sortie en mode batch (défaut : predictions_llm.parquet).",
    )
    parser.add_argument(
        "--concurrency", type=int, default=20,
        help="Nombre d'appels API simultanés en mode batch (défaut : 20).",
    )
    parser.add_argument(
        "--full-nomenclature", action="store_true",
        help=(
            "Envoie les 700 codes à chaque requête au lieu de filtrer "
            "sur les sections prédites. Plus lent et plus coûteux."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Niveau de log console (défaut : info).",
    )
    parser.add_argument(
        "--log-file", default=None,
        help=(
            "Fichier de log (DEBUG complet). "
            "Défaut : decide_coicop.log dans le même dossier que --output-file."
        ),
    )
    args = parser.parse_args()

    # ── Logging ──
    log_file: Path | None
    if args.log_file:
        log_file = Path(args.log_file)
    else:
        log_file = Path(args.output_file).with_suffix(".log")
    setup_logging(args.log_level, log_file)

    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("Variable OPENAI_API_KEY non définie.")
        sys.exit(1)

    logger.info("Chargement des données...")
    df = load_all_observations()

    logger.info("Chargement de la nomenclature...")
    nomenclature = load_nomenclature()

    # ── Mode observation unique ──
    if args.id is not None:
        obs = get_observation(df, args.id)
        prompt = build_prompt(obs, nomenclature, full_nomen=args.full_nomenclature)

        api_kwargs: dict = {"api_key": os.environ["OPENAI_API_KEY"]}
        if base_url := os.environ.get("OPENAI_BASE_URL"):
            api_kwargs["base_url"] = base_url
            logger.debug("OPENAI_BASE_URL=%s", base_url)
        client = OpenAI(**api_kwargs)

        logger.info("Appel au modèle %s pour id=%s (nomen=%s)...",
                    args.model, args.id,
                    "complète" if args.full_nomenclature else "filtrée")
        decision = call_llm_sync(prompt, args.model, client)

        if args.output == "json":
            print(json.dumps({
                "id":             args.id,
                "raw_product":    obs.get("raw_product"),
                "shop":           obs.get("shop"),
                "shop_type_name": obs.get("shop_type_name"),
                "budget":         obs.get("budget"),
                "code_reference": obs.get("code"),
                "coicop_code":    decision.coicop_code,
                "libelle":        decision.libelle,
                "explication":    decision.explication,
                "confiance":      decision.confiance,
            }, ensure_ascii=False, indent=2))
        else:
            print_result(obs, decision)
        return

    # ── Mode batch ──
    asyncio.run(run_batch(
        df=df,
        nomenclature=nomenclature,
        model=args.model,
        concurrency=args.concurrency,
        output_file=Path(args.output_file),
        full_nomen=args.full_nomenclature,
    ))


if __name__ == "__main__":
    main()
