/**
 * clinical_summary.js — Natural language clinical summary renderer.
 */

function renderClinicalSummary() {
    const el = document.getElementById('clinical-summary');

    if (!patientData) {
        el.textContent = 'Loading clinical summary...';
        return;
    }

    let summary = '';

    // Basic demographics
    const age = patientData.age_at_baseline?.toFixed(0) || '?';
    const sex = patientData.sex === 1 ? 'female' : 'male';
    const stage = patientData.current_stage || '?';
    const nVisits = patientData.n_visits || 0;
    const followup = patientData.follow_up_months?.toFixed(0) || '?';

    summary += `Patient ${PATNO} is a ${age}-year-old ${sex} currently at NSD-ISS Stage ${stage}`;
    summary += `, with ${nVisits} visits over ${followup} months of follow-up. `;

    // Trajectory summary
    const trajectory = patientData.stage_trajectory || [];
    if (trajectory.length > 1) {
        const uniqueStages = [...new Set(trajectory)];
        const initialStage = trajectory[0];
        const finalStage = trajectory[trajectory.length - 1];

        if (initialStage !== finalStage) {
            summary += `Disease trajectory: ${initialStage} to ${finalStage} `;
            summary += `(${uniqueStages.length} distinct stages visited). `;
        } else {
            summary += `Stage has remained stable at ${finalStage} throughout follow-up. `;
        }
    }

    // Genetics
    const genetics = [];
    if (patientData.lrrk2_carrier) genetics.push('LRRK2');
    if (patientData.gba_carrier) genetics.push('GBA');
    if (genetics.length > 0) {
        summary += `Genetic risk: ${genetics.join(', ')} carrier. `;
    }

    // Staging prediction
    if (stagingData) {
        summary += `CatBoost staging prediction: Stage ${stagingData.predicted_stage} `;
        summary += `(${(Math.max(...Object.values(stagingData.probabilities)) * 100).toFixed(0)}% confidence). `;
    }

    // Survival prediction
    if (survivalData && survivalData.top_transitions?.length > 0) {
        const top = survivalData.top_transitions[0];
        summary += `Most likely transition: Stage ${top.destination_stage} `;
        summary += `(${(top.cif_at_12mo * 100).toFixed(1)}% at 1yr, `;
        summary += `${(top.cif_at_36mo * 100).toFixed(1)}% at 3yr). `;

        if (survivalData.conformal_band_width) {
            summary += `90% conformal band width: ${survivalData.conformal_band_width.toFixed(3)}. `;
        }
    }

    // Gate activation
    if (survivalData?.gate_activation != null) {
        const gatePct = (survivalData.gate_activation * 100).toFixed(1);
        const tempPct = ((1 - survivalData.gate_activation) * 100).toFixed(1);
        summary += `Graph-DT fusion: ${tempPct}% temporal, ${gatePct}% graph context. `;
    }

    el.textContent = summary;
}

// Run after a short delay to allow other panels to load first
setTimeout(renderClinicalSummary, 2000);
