import os
import sys
import json
import re
from collections import Counter
import numpy as np
from sentence_transformers import SentenceTransformer

SECTIONS = { 
    1: "cement and concrete", 2: "building limes", 3: "stones", 
    4: "wood products for building", 5: "gypsum building materials", 
    6: "timber", 7: "bitumen and tar products", 8: "floor wall roof coverings finishes", 
    9: "waterproofing damp proofing", 10: "sanitary appliances water fittings", 
    11: "builders hardware", 12: "wood products", 13: "doors windows shutters", 
    14: "concrete reinforcement", 15: "structural steels", 16: "light metal alloys", 
    17: "structural shapes", 18: "welding electrodes wires", 19: "threaded fasteners rivets", 
    20: "wire ropes wire products", 21: "glass", 22: "fillers stoppers putties", 
    23: "thermal insulation materials", 24: "plastics", 25: "conductors cables", 
    26: "wiring accessories", 27: "general" 
}

def normalize_key(std):
    base = re.sub(r"\s+", " ", str(std.get("id", "")).strip().upper())
    part = std.get("part")
    if part:
        cleaned = re.sub(r"\s+", " ", str(part).strip())
        m = re.match(r"^(?:PART)\s*(.*)$", cleaned, flags=re.IGNORECASE)
        suffix = m.group(1).strip() if m else cleaned
        part_label = f"Part {suffix.upper()}" if suffix else None
        if part_label: return f"{base} ({part_label})"
    return base

def main():
    print("Loading BAAI/bge-m3 model...")
    model = SentenceTransformer("BAAI/bge-m3")
    with open("data/sp21_standards.json", "r", encoding="utf-8") as f: standards = json.load(f)
    
    section_ids = list(SECTIONS.keys())
    section_embs = model.encode([SECTIONS[sid] for sid in section_ids], normalize_embeddings=True)
    
    standard_to_section = {}
    section_assignments = {sid: [] for sid in section_ids}
    
    print("Assigning standards...")
    for std in standards:
        std_key = normalize_key(std)
        text = f"{std.get('title', '')} {std.get('content', '')[:300]}"
        std_emb = model.encode(text, normalize_embeddings=True)
        sims = np.dot(section_embs, std_emb)
        best_sid = section_ids[np.argmax(sims)]
        if "TILE" in std.get('title', '').upper(): best_sid = 8
        if np.max(sims) < 0.15: best_sid = 27
        standard_to_section[std_key] = {"section_id": best_sid}
        section_assignments[best_sid].append(std)
        
    print("Clustering sub-domains...")
    section_profiles = {}
    for sid, stds in section_assignments.items():
        # Simple clustering: Top 3 keywords that aren't the section name
        titles = [s.get('title', '').lower() for s in stds]
        all_words = re.findall(r'\b[a-z]{4,}\b', " ".join(titles))
        stop = {"specification", "requirements", "general", "purposes", "with", "this", "covers"}
        # Filter out section name words
        sec_words = set(SECTIONS[sid].lower().split())
        sub_domains = []
        counts = Counter(all_words)
        for w, c in counts.most_common(20):
            if w not in stop and w not in sec_words:
                sub_domains.append({"name": w, "standards": [normalize_key(s) for s in stds if w in s.get('title', '').lower()][:10]})
                if len(sub_domains) >= 3: break
        
        section_profiles[str(sid)] = {
            "id": sid, "name": SECTIONS[sid], 
            "sub_domains": sub_domains,
            "core_standards": [normalize_key(s) for s in stds[:10]]
        }
        
    with open("data/standard_to_section.json", "w", encoding="utf-8") as f: json.dump(standard_to_section, f, indent=2)
    with open("data/section_profiles.json", "w", encoding="utf-8") as f: json.dump(section_profiles, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    main()
