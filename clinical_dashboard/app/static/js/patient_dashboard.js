/**
 * patient_dashboard.js — Orchestrates all panels for one patient.
 * Loads patient data and initializes each panel.
 */

let patientData = null;
let stagingData = null;
let survivalData = null;

async function loadPatientDashboard() {
    try {
        // 1. Load patient detail
        const detailResp = await fetch(`/api/patients/${PATNO}`);
        if (!detailResp.ok) {
            document.getElementById('clinical-summary').textContent =
                `Patient ${PATNO} not found.`;
            return;
        }
        patientData = await detailResp.json();

        // Populate header
        populateHeader(patientData);

        // 2. Load all panels in parallel
        const promises = [
            renderTrajectoryChart(patientData),
            loadStagingPanel(PATNO),
            loadSurvivalPanel(PATNO),
            loadAnnotationsPanel(PATNO),
            loadSimilarityPanel(PATNO),
            loadWearablePanel(PATNO),
        ];

        await Promise.allSettled(promises);

        // 3. Initialize what-if sliders (needs staging data)
        initWhatIfSliders(patientData);

    } catch (err) {
        console.error('Dashboard load error:', err);
        document.getElementById('clinical-summary').textContent =
            'Error loading patient data. Please try again.';
    }
}

function populateHeader(data) {
    const stage = data.current_stage;
    const color = STAGE_COLORS[stage] || '#7f8c8d';

    // Stage badge
    const badge = document.getElementById('stage-badge');
    badge.style.backgroundColor = color;
    badge.textContent = stage;

    // Text info
    document.getElementById('header-age').textContent = data.age_at_baseline?.toFixed(1) ?? '--';
    document.getElementById('header-sex').textContent = data.sex === 1 ? 'Female' : 'Male';
    document.getElementById('header-visits').textContent = data.n_visits;
    document.getElementById('header-followup').textContent = data.follow_up_months?.toFixed(0) ?? '--';
    document.getElementById('header-stage').textContent = stage;
    document.getElementById('header-stage').style.color = color;
    document.getElementById('header-stage-label').textContent =
        STAGE_DESCRIPTIONS[stage] || '';

    // Genetics
    const genetics = [];
    if (data.lrrk2_carrier) genetics.push('LRRK2+');
    if (data.gba_carrier) genetics.push('GBA+');
    document.getElementById('header-genetics').textContent =
        genetics.length ? genetics.join(' / ') : 'No genetic risk variants';
}

// Start loading when page is ready
document.addEventListener('DOMContentLoaded', loadPatientDashboard);
