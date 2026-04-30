const CATEGORY_ICONS = {
    1: "🧱",
    2: "🏗️",
    3: "🪨",
    4: "🪵",
    5: "💎",
    6: "🌲",
    7: "🛢️",
    8: "🏠",
    9: "💧",
    10: "🚽",
    11: "🗝️",
    12: "🧺",
    13: "🚪",
    14: "⛓️",
    15: "⚙️",
    16: "💡",
    17: "📐",
    18: "🔥",
    19: "🔩",
    20: "🪢",
    21: "✨",
    22: "🧴",
    23: "🌡️",
    24: "🧪",
    25: "⚡",
    26: "🔌",
    27: "🛠️"
};

const BASE_PROMPTS = [
    "High-strength Portland slag cement for marine environment",
    "33 grade ordinary Portland cement for masonry blocks",
    "Copper conductors for domestic wiring harness",
    "Waterproofing membrane for rooftop applications",
    "Structural steel sections for warehouse construction",
    "Wood products for interior joinery and doors"
];

const SECTION_11_FALLBACK = {
    keywords: ["locks", "hinges", "bolts", "door closers", "fasteners"],
    standards: [
        "IS 205",
        "IS 206",
        "IS 208",
        "IS 1019",
        "IS 2209",
        "IS 3847",
        "IS 4621",
        "IS 4992",
        "IS 5187",
        "IS 5899"
    ]
};

let selectedCategoryId = null;
let guidedProfiles = {};
let guidedMetadata = {};
let promptTimer = null;
let promptBank = [...BASE_PROMPTS];
let promptState = { text: "", index: 0, deleting: false, hold: 0 };
let loadingTimer = null;
let loadingPhaseIndex = 0;

document.addEventListener("DOMContentLoaded", async () => {
    await loadGuidedData();
    initSearch();
    startPromptRotation(promptBank);
});

async function loadGuidedData() {
    try {
        const response = await fetch("/api/guided_data");
        const data = await response.json();
        guidedProfiles = data.profiles || {};
        guidedMetadata = data.metadata || {};
        renderCategoryGrid();
    } catch (error) {
        const grid = document.getElementById("category-grid");
        grid.innerHTML = `<div class="empty-state">Unable to load guided sections: ${error.message}</div>`;
    }
}

function renderCategoryGrid() {
    const grid = document.getElementById("category-grid");
    const ids = Object.keys(guidedProfiles).map(Number).sort((a, b) => a - b);
    grid.innerHTML = "";

    ids.forEach(id => {
        const profile = guidedProfiles[String(id)];
        if (!profile) return;
        const item = document.createElement("div");
        item.className = "category-item";
        item.dataset.sectionId = String(id);
        item.innerHTML = `
            <span class="category-icon">${CATEGORY_ICONS[id] || "🧩"}</span>
            <span class="category-name">${titleCase(profile.name || `Section ${id}`)}</span>
        `;
        item.onclick = () => selectSection(String(id), item);
        grid.appendChild(item);
    });
}

function titleCase(text) {
    return String(text)
        .replace(/\s+/g, " ")
        .trim()
        .split(" ")
        .map(word => word ? word[0].toUpperCase() + word.slice(1) : word)
        .join(" ");
}

function selectSection(id, element) {
    document.querySelectorAll(".category-item").forEach(el => el.classList.remove("active"));
    element.classList.add("active");
    selectedCategoryId = id;

    const section = guidedProfiles[id];
    const keywords = getSectionKeywords(section);
    const examplesHint = document.getElementById("examples-hint");
    examplesHint.innerHTML = '<b>Try these:</b> ' + keywords.slice(0, 4).map(keyword => `<span class="example-tag" onclick="useExample('${escapeHtml(keyword)}')">${keyword}</span>`).join(" ");

    renderKeywords(id, keywords);
    updatePromptBank(section, keywords);
}

function getSectionKeywords(section) {
    if (!section) return [];
    const keywords = [];
    (section.sub_domains || []).forEach(domain => {
        if (domain && domain.name) keywords.push(domain.name);
    });
    if (Array.isArray(section.core_standards) && section.core_standards.length) {
        keywords.push("core standards");
    }
    if (keywords.length === 0 && String(section?.id) === "11") {
        return SECTION_11_FALLBACK.keywords;
    }
    return [...new Set(keywords)].slice(0, 8);
}

function renderKeywords(sectionId, keywords) {
    const keywordsContainer = document.getElementById("keywords-container");
    const keywordsList = document.getElementById("keywords-list");
    const guidedResultsContainer = document.getElementById("guided-results-container");
    const guidedResultsList = document.getElementById("guided-results-list");

    keywordsContainer.style.display = "block";
    guidedResultsContainer.style.display = "none";
    keywordsList.innerHTML = "";
    guidedResultsList.innerHTML = "";

    const section = guidedProfiles[sectionId];
    const standardBuckets = buildStandardBuckets(section);

    keywords.forEach(keyword => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "keyword-chip";
        button.textContent = keyword;
        button.onclick = () => renderGuidedResults(keyword, standardBuckets[keyword] || []);
        keywordsList.appendChild(button);
    });

    if (keywords.length) {
        renderGuidedResults(keywords[0], standardBuckets[keywords[0]] || []);
    }
}

function buildStandardBuckets(section) {
    const buckets = {};
    if (!section) return buckets;

    (section.sub_domains || []).forEach(domain => {
        const keyword = domain?.name || "";
        buckets[keyword] = (domain?.standards || []).map(code => ({
            code,
            info: getStandardInfo(code)
        }));
    });

    buckets["core standards"] = (section.core_standards || []).map(code => ({
        code,
        info: getStandardInfo(code)
    }));

    if (String(section.id) === "11" && Object.keys(buckets).length <= 1) {
        const hardwareStandards = SECTION_11_FALLBACK.standards.map(code => ({
            code,
            info: getStandardInfo(code)
        }));
        const byKeyword = {
            locks: ["IS 2209", "IS 3847", "IS 1019", "IS 9131", "IS 7540"],
            hinges: ["IS 205", "IS 206", "IS 362", "IS 453", "IS 9106", "IS 12817"],
            bolts: ["IS 281", "IS 5187", "IS 7534", "IS 10019"],
            "door closers": ["IS 3564", "IS 6343", "IS 14912", "IS 6315", "IS 7197"],
            fasteners: ["IS 1120", "IS 1365", "IS 1366", "IS 3757", "IS 6113"]
        };
        Object.entries(byKeyword).forEach(([keyword, codes]) => {
            buckets[keyword] = codes.map(code => ({ code, info: getStandardInfo(code) }));
        });
        buckets["core standards"] = hardwareStandards;
    }

    return buckets;
}

function getStandardInfo(code) {
    const record = guidedMetadata[code] || {};
    const title = String(record.title || "BIS standard").replace(/\s+/g, " ").trim();
    return title.length > 88 ? `${title.slice(0, 85)}...` : title;
}

function renderGuidedResults(keyword, items) {
    const guidedResultsContainer = document.getElementById("guided-results-container");
    const guidedResultsList = document.getElementById("guided-results-list");
    guidedResultsContainer.style.display = "block";
    guidedResultsList.innerHTML = "";

    const header = document.createElement("div");
    header.className = "keyword-results-header";
    header.innerHTML = `<span>Keyword:</span> <strong>${keyword}</strong>`;
    guidedResultsList.appendChild(header);

    if (!items.length) {
        const empty = document.createElement("div");
        empty.className = "mini-empty";
        empty.textContent = "No standards found for this keyword.";
        guidedResultsList.appendChild(empty);
        return;
    }

    items.forEach(item => {
        const card = document.createElement("div");
        card.className = "guided-result-card";
        card.innerHTML = `
            <div class="guided-code">${item.code}</div>
            <div class="guided-info">${item.info}</div>
        `;
        guidedResultsList.appendChild(card);
    });
}

function updatePromptBank(section, keywords) {
    const sectionName = titleCase(section?.name || "Product");
    const lead = keywords[0] || sectionName;
    const sub = keywords[1] || keywords[0] || sectionName;
    const tertiary = keywords[2] || sectionName;

    promptBank = [
        `${sectionName} for industrial use`,
        `${lead} for commercial construction`,
        `${sub} for BIS compliant manufacturing`,
        `High-strength ${sectionName.toLowerCase()} for heavy duty use`,
        `${tertiary} used in building and infrastructure`,
        `Best BIS standard for ${sectionName.toLowerCase()} applications`
    ];

    resetPromptRotation();
}

function startPromptRotation(prompts) {
    promptBank = prompts.slice(0, 6);
    const input = document.getElementById("query-input");
    if (!input) return;

    stopPromptRotation();
    promptState = { text: "", index: 0, deleting: false, hold: 0 };
    input.setAttribute("placeholder", "");

    const tick = () => {
        const fullText = promptBank[promptState.index % promptBank.length] || BASE_PROMPTS[0];

        if (promptState.deleting) {
            promptState.text = fullText.slice(0, Math.max(0, promptState.text.length - 1));
            if (!promptState.text) {
                promptState.deleting = false;
                promptState.index = (promptState.index + 1) % promptBank.length;
                promptState.hold = 0;
            }
        } else {
            promptState.text = fullText.slice(0, promptState.text.length + 1);
            if (promptState.text === fullText) {
                promptState.hold += 1;
                if (promptState.hold > 10) {
                    promptState.deleting = true;
                }
            }
        }

        input.setAttribute("placeholder", promptState.text || fullText);
    };

    tick();
    promptTimer = setInterval(tick, 55);
}

function resetPromptRotation() {
    startPromptRotation(promptBank);
}

function stopPromptRotation() {
    if (promptTimer) {
        clearInterval(promptTimer);
        promptTimer = null;
    }
}

function useExample(text) {
    document.getElementById("query-input").value = text;
}

function initSearch() {
    const btn = document.getElementById("search-btn");
    const input = document.getElementById("query-input");

    btn.onclick = performSearch;
    input.onkeydown = (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            performSearch();
        }
    };
}

async function performSearch() {
    const input = document.getElementById("query-input");
    const query = input.value.trim();
    const btn = document.getElementById("search-btn");
    const resultsList = document.getElementById("results-list");

    if (!query) return;

    btn.disabled = true;
    btn.querySelector(".btn-text").textContent = "Searching...";
    resultsList.innerHTML = renderLoadingState();
    startLoadingAnimation();
    document.querySelector(".search-section")?.scrollIntoView({ behavior: "smooth", block: "start" });
    document.getElementById("results-section")?.scrollIntoView({ behavior: "smooth", block: "start" });

    try {
        const response = await fetch("/api/search", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query, category_id: selectedCategoryId })
        });

        const data = await response.json();
        stopLoadingAnimation();
        renderResults(data);
        document.getElementById("results-section")?.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
        stopLoadingAnimation();
        resultsList.innerHTML = `<div class="empty-state" style="color: #ef4444;">Error: ${error.message}</div>`;
    } finally {
        btn.disabled = false;
        btn.querySelector(".btn-text").textContent = "Search via AI Agent";
    }
}

function renderLoadingState() {
    return `
        <div class="loading-state">
            <div class="loading-top">
                <div class="spinner"></div>
                <div>
                    <div class="loading-title">🧠 Thinking and analyzing standards...</div>
                    <div class="loading-subtitle">Searching BIS records and ranking the best matches.</div>
                </div>
            </div>
            <div class="loading-phases">
                <span class="loading-phase active">Scanning index</span>
                <span class="loading-phase">Matching sections</span>
                <span class="loading-phase">Ranking standards</span>
                <span class="loading-phase">Writing summary</span>
            </div>
            <div class="loading-track">
                <div class="loading-bar"></div>
            </div>
        </div>
    `;
}

function startLoadingAnimation() {
    const phases = ["Scanning index", "Matching sections", "Ranking standards", "Writing summary"];
    const phaseNodes = Array.from(document.querySelectorAll(".loading-phase"));
    const title = document.querySelector(".loading-title");
    let step = 0;

    stopLoadingAnimation();
    loadingPhaseIndex = 0;

    loadingTimer = setInterval(() => {
        if (!phaseNodes.length) return;
        phaseNodes.forEach((node, index) => node.classList.toggle("active", index === loadingPhaseIndex));
        if (title) title.textContent = `🧠 ${phases[loadingPhaseIndex]}...`;
        loadingPhaseIndex = (loadingPhaseIndex + 1) % phases.length;
    }, 700);
}

function stopLoadingAnimation() {
    if (loadingTimer) {
        clearInterval(loadingTimer);
        loadingTimer = null;
    }
}

function renderResults(data) {
    const resultsList = document.getElementById("results-list");
    const standards = data.retrieved || [];
    const rationale = data.rationale || "";
    const latency = Number(data.latency_seconds || 0);
    const realLatency = latency.toFixed(2);

    if (standards.length === 0) {
        resultsList.innerHTML = '<div class="empty-state">No standards found. Try broadening your description.</div>';
        return;
    }

    resultsList.innerHTML = `
        <div class="summary-card">
            <div class="summary-header">
                <div>
                    <div class="summary-kicker">AI Summary</div>
                    <h3 class="summary-title">Best BIS matches for your query</h3>
                </div>
                <div class="summary-latency">Achieved latency ${realLatency}s</div>
            </div>
            <div class="summary-body">
                <p class="summary-text">${rationale}</p>
                <div class="summary-label">Recommended standards</div>
                <div class="summary-standards">
                    ${standards.slice(0, 5).map((std, index) => `<span class="summary-chip${index === 0 ? ' summary-chip-best' : ''}">${std}</span>`).join("")}
                </div>
            </div>
        </div>
    `;
}

function escapeHtml(text) {
    return String(text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text);
}
