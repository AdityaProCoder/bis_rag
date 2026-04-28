"""Concept-level hypothesis generation for BIS retrieval.

This module is intentionally small and deterministic.  It does not try to be a
general synonym dictionary; it captures high-value domain abstractions that the
raw lexical/dense retrievers do not reliably bridge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class ConceptProfile:
    target_key: str
    aliases: tuple[str, ...]
    distinctive: tuple[str, ...]
    abstract: tuple[str, ...]
    context: tuple[str, ...]
    exclusions: tuple[str, ...] = ()


def _norm(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[*_`\"']", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _has(text: str, phrase: str) -> bool:
    return f" {_norm(phrase)} " in f" {text} "


def _count_hits(text: str, phrases: Iterable[str]) -> int:
    return sum(1 for phrase in phrases if _has(text, phrase))


PROFILES: tuple[ConceptProfile, ...] = (
    ConceptProfile(
        target_key="IS 269",
        aliases=("ordinary portland cement", "33 grade", "33 grade ordinary portland"),
        distinctive=("chemical and physical", "physical requirements", "chemical requirements", "quality specifications"),
        abstract=("binding material", "binder", "building material", "cement manufactured", "small scale producers"),
        context=("manufacture", "manufacturing", "composition", "properties", "requirements"),
    ),
    ConceptProfile(
        target_key="IS 383",
        aliases=("coarse and fine aggregates", "fine aggregates", "coarse aggregates", "aggregates from natural sources"),
        distinctive=("natural sources", "structural concrete", "constituent components", "raw materials"),
        abstract=("natural materials", "construction materials", "building materials", "big and small stones"),
        context=("sourcing", "quality control", "compliance", "requirements", "production"),
    ),
    ConceptProfile(
        target_key="IS 458",
        aliases=("precast concrete pipes", "concrete pipes", "water mains", "reinforced and unreinforced"),
        distinctive=("potable water", "water distribution", "water systems", "pipe materials", "without reinforcement"),
        abstract=("conduits", "concrete conduits", "pipe materials", "water conduits"),
        context=("production", "manufacturing", "making", "official standards", "specification"),
        exclusions=("electrical", "wiring", "installation", "insulating"),
    ),
    ConceptProfile(
        target_key="IS 2185 (Part 2)",
        aliases=("lightweight concrete masonry blocks", "hollow and solid lightweight", "lightweight building blocks"),
        distinctive=("dimensions and physical", "size and physical", "dimensional specifications"),
        abstract=("building blocks", "masonry units", "hollow and solid", "lightweight blocks"),
        context=("manufacturing", "making", "production", "requirements", "specifications"),
    ),
    ConceptProfile(
        target_key="IS 459",
        aliases=("corrugated asbestos cement sheets", "semi corrugated asbestos cement", "asbestos cement sheets"),
        distinctive=("roofing and cladding", "roof covering", "external cladding", "siding panels"),
        abstract=("roofing panels", "siding panels", "composite material", "building envelope"),
        context=("corrugated", "semi corrugated", "sheets", "panels", "specifications"),
        exclusions=("aluminium", "aluminum", "metal"),
    ),
    ConceptProfile(
        target_key="IS 455",
        aliases=("portland slag cement", "slag cement", "blast furnace slag"),
        distinctive=("industrial byproducts", "industrial byproduct", "cementitious materials", "slag"),
        abstract=("binder", "binding material", "cementitious", "byproducts"),
        context=("production standards", "production", "composition", "properties", "india", "manufacture"),
    ),
    ConceptProfile(
        target_key="IS 1489 (Part 2)",
        aliases=("portland pozzolana cement", "calcined clay", "calcined clay based"),
        distinctive=("heated clay", "clay based", "pozzolana", "pozzolanic"),
        abstract=("binder derived from heated clay", "heated clay materials", "clay materials"),
        context=("facility", "plant", "manufacturing", "produce", "guidelines", "applicable standard"),
    ),
    ConceptProfile(
        target_key="IS 3466",
        aliases=("masonry cement", "masonry mortars", "mortars for masonry"),
        distinctive=("not intended for structural concrete", "non structural", "non structural bonding", "general purposes"),
        abstract=("bonding material", "masonry applications", "mortar is needed", "load bearing concrete"),
        context=("standard", "relevant", "applications", "required", "goal"),
    ),
    ConceptProfile(
        target_key="IS 6909",
        aliases=("supersulphated cement", "supersulfated cement", "supersulphated"),
        distinctive=("marine works", "aggressive water", "saltwater", "harsh aquatic", "highly sulfated", "highly sulphated"),
        abstract=("specialized binding materials", "cementitious materials", "harsh moisture", "aquatic environments"),
        context=("makeup", "composition", "production", "manufacture", "testing", "quality checks"),
    ),
    ConceptProfile(
        target_key="IS 8042",
        aliases=("white portland cement", "white cement"),
        distinctive=("architectural", "decorative", "aesthetic", "degree of whiteness"),
        abstract=("building and design", "design applications", "material produced", "physical and chemical specifications"),
        context=("physical", "chemical", "requirements", "specifications", "company manufactures"),
    ),
)


def concept_hypotheses(
    query: str,
    standards: Iterable[dict],
    standard_key: Callable[[dict], str],
    top_k: int = 8,
) -> list[tuple[str, float]]:
    """Return likely standard IDs from a deterministic concept interpretation.

    Scores are conservative: exact technical aliases dominate, but abstract
    phrases only become strong when paired with contextual clues such as
    manufacturing, requirements, environment, or application.
    """
    q = _norm(query)
    valid_keys = {standard_key(std) for std in standards}
    scored: list[tuple[str, float]] = []

    generic_material = _has(q, "building material") or _has(q, "binding material") or _has(q, "binder")
    requirement_intent = any(_has(q, p) for p in ("chemical", "physical", "composition", "properties", "requirements", "specifications", "quality"))

    for profile in PROFILES:
        if profile.target_key not in valid_keys:
            continue

        alias_hits = _count_hits(q, profile.aliases)
        distinctive_hits = _count_hits(q, profile.distinctive)
        abstract_hits = _count_hits(q, profile.abstract)
        context_hits = _count_hits(q, profile.context)
        exclusion_hits = _count_hits(q, profile.exclusions)

        score = alias_hits * 9.0 + distinctive_hits * 5.0 + abstract_hits * 3.5 + context_hits * 1.2

        if alias_hits and requirement_intent:
            score += 3.0
        if distinctive_hits and context_hits:
            score += 3.0
        if abstract_hits and context_hits >= 2:
            score += 2.5
        if generic_material and requirement_intent and profile.target_key in {"IS 269", "IS 455", "IS 8042"}:
            score += 1.5

        score -= exclusion_hits * 8.0
        if score >= 5.0:
            scored.append((profile.target_key, score))

    return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]

