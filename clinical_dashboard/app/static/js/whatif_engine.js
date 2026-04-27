/**
 * whatif_engine.js — Feature sliders, presets, debounced re-prediction.
 */

const FEATURE_RANGES = {
    AGE_AT_BASELINE:      { min: 30, max: 90, step: 1, label: 'Age at Baseline' },
    SEX:                  { min: 0, max: 1, step: 1, label: 'Sex (0=M, 1=F)' },
    UPDRS1_TOTAL:         { min: 0, max: 52, step: 1, label: 'UPDRS-I (Non-Motor)' },
    UPDRS2_TOTAL:         { min: 0, max: 52, step: 1, label: 'UPDRS-II (Motor ADL)' },
    UPDRS3_TREMOR:        { min: 0, max: 40, step: 1, label: 'UPDRS-III Tremor' },
    UPDRS3_RIGIDITY:      { min: 0, max: 20, step: 1, label: 'UPDRS-III Rigidity' },
    UPDRS3_BRADYKINESIA:  { min: 0, max: 36, step: 1, label: 'UPDRS-III Bradykinesia' },
    UPDRS3_AXIAL:         { min: 0, max: 20, step: 1, label: 'UPDRS-III Axial' },
    UPDRS4_TOTAL:         { min: 0, max: 24, step: 1, label: 'UPDRS-IV (Complications)' },
    MOCA_TOTAL:           { min: 0, max: 30, step: 1, label: 'MoCA (Cognitive)' },
    ESS_TOTAL:            { min: 0, max: 24, step: 1, label: 'Epworth Sleepiness' },
    RBD_TOTAL:            { min: 0, max: 13, step: 1, label: 'RBD Screening' },
};

let baselineFeatures = {};
let modifiedFeatures = {};
let whatifTimeout = null;

function initWhatIfSliders(patientData) {
    const container = document.getElementById('whatif-sliders');
    container.innerHTML = '';

    // Get baseline features from staging data (if available)
    if (stagingData && stagingData.features_used) {
        baselineFeatures = { ...stagingData.features_used };
    } else {
        // Fallback: empty features
        Object.keys(FEATURE_RANGES).forEach(f => { baselineFeatures[f] = 0; });
    }

    modifiedFeatures = { ...baselineFeatures };

    // Create a slider for each feature
    Object.entries(FEATURE_RANGES).forEach(([feat, range]) => {
        const currentVal = baselineFeatures[feat] ?? range.min;

        const row = document.createElement('div');
        row.className = 'flex items-center gap-2';
        row.innerHTML = `
            <label class="text-xs text-clinical-600 w-32 flex-shrink-0 truncate" title="${range.label}">
                ${range.label}
            </label>
            <input type="range" id="slider-${feat}" data-feature="${feat}"
                   min="${range.min}" max="${range.max}" step="${range.step}" value="${currentVal}"
                   class="flex-1">
            <span class="text-xs font-mono text-clinical-700 w-8 text-right" id="val-${feat}">
                ${Math.round(currentVal)}
            </span>
        `;
        container.appendChild(row);

        // Slider change handler
        const slider = row.querySelector(`#slider-${feat}`);
        slider.addEventListener('input', (e) => {
            const newVal = parseFloat(e.target.value);
            document.getElementById(`val-${feat}`).textContent = Math.round(newVal);
            modifiedFeatures[feat] = newVal;

            // Visual feedback for modified sliders
            if (Math.abs(newVal - (baselineFeatures[feat] ?? 0)) > 0.01) {
                slider.classList.add('modified');
            } else {
                slider.classList.remove('modified');
            }

            // Debounced what-if request
            clearTimeout(whatifTimeout);
            whatifTimeout = setTimeout(runWhatIf, 300);
        });
    });
}

async function runWhatIf() {
    // Build modifications list (only changed features)
    const modifications = [];
    Object.entries(modifiedFeatures).forEach(([feat, val]) => {
        if (Math.abs(val - (baselineFeatures[feat] ?? 0)) > 0.01) {
            modifications.push({ feature: feat, new_value: val });
        }
    });

    if (modifications.length === 0) {
        document.getElementById('whatif-result').classList.add('hidden');
        return;
    }

    try {
        const resp = await fetch('/api/whatif/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ patno: PATNO, modifications }),
        });

        if (!resp.ok) return;
        const result = await resp.json();

        // Show result
        const resultDiv = document.getElementById('whatif-result');
        resultDiv.classList.remove('hidden');

        const explDiv = document.getElementById('whatif-explanation');
        explDiv.innerHTML = '';

        if (result.staging_changed) {
            explDiv.innerHTML += `<div class="text-red-600 font-medium mb-1">Stage changed:
                ${result.baseline_staging?.predicted_stage} &rarr; ${result.counterfactual_staging?.predicted_stage}
            </div>`;
        }
        explDiv.innerHTML += `<pre class="whitespace-pre-wrap text-xs">${result.explanation}</pre>`;

        // Update CIF chart with counterfactual overlay
        if (survivalData && result.counterfactual_survival) {
            renderCIFChart(survivalData, result.counterfactual_survival);
        }

    } catch (err) {
        console.error('What-if error:', err);
    }
}

function applyPreset(preset) {
    const changes = {
        medication: {
            // Note: pdmedyn isn't in the 12-feature CatBoost model, but this is
            // a demonstration of the concept. In practice, medication effects would
            // be modeled through their clinical impacts.
            UPDRS3_TREMOR: Math.max(0, (baselineFeatures.UPDRS3_TREMOR ?? 5) - 3),
            UPDRS3_RIGIDITY: Math.max(0, (baselineFeatures.UPDRS3_RIGIDITY ?? 5) - 2),
            UPDRS3_BRADYKINESIA: Math.max(0, (baselineFeatures.UPDRS3_BRADYKINESIA ?? 5) - 3),
        },
        motor_decline: {
            UPDRS3_TREMOR: Math.min(40, (baselineFeatures.UPDRS3_TREMOR ?? 5) + 5),
            UPDRS3_RIGIDITY: Math.min(20, (baselineFeatures.UPDRS3_RIGIDITY ?? 5) + 3),
            UPDRS3_BRADYKINESIA: Math.min(36, (baselineFeatures.UPDRS3_BRADYKINESIA ?? 5) + 5),
            UPDRS3_AXIAL: Math.min(20, (baselineFeatures.UPDRS3_AXIAL ?? 3) + 3),
        },
        cognitive_decline: {
            MOCA_TOTAL: Math.max(0, (baselineFeatures.MOCA_TOTAL ?? 26) - 3),
            UPDRS1_TOTAL: Math.min(52, (baselineFeatures.UPDRS1_TOTAL ?? 5) + 4),
        },
    };

    const preset_changes = changes[preset];
    if (!preset_changes) return;

    Object.entries(preset_changes).forEach(([feat, val]) => {
        modifiedFeatures[feat] = val;
        const slider = document.getElementById(`slider-${feat}`);
        if (slider) {
            slider.value = val;
            slider.classList.add('modified');
            document.getElementById(`val-${feat}`).textContent = Math.round(val);
        }
    });

    clearTimeout(whatifTimeout);
    whatifTimeout = setTimeout(runWhatIf, 100);
}

function resetWhatIf() {
    modifiedFeatures = { ...baselineFeatures };

    Object.entries(baselineFeatures).forEach(([feat, val]) => {
        const slider = document.getElementById(`slider-${feat}`);
        if (slider) {
            slider.value = val;
            slider.classList.remove('modified');
            document.getElementById(`val-${feat}`).textContent = Math.round(val);
        }
    });

    document.getElementById('whatif-result').classList.add('hidden');

    // Reset CIF chart to baseline
    if (survivalData) {
        renderCIFChart(survivalData);
    }
}
