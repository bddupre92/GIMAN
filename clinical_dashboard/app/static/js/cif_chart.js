/**
 * cif_chart.js — Plotly.js CIF curves with conformal bands.
 * Shows DeepHit (solid) and Graph-DT (dashed) for top transitions.
 */

const TIME_BINS = [3, 6, 12, 18, 24, 36, 48, 60, 84, 120, 180];
const CAUSE_LABELS = { 0: '0', 1: '1', 2: '2B', 3: '3', 4: '4', 5: '5', 6: '6' };

async function loadSurvivalPanel(patno) {
    try {
        const resp = await fetch('/api/survival/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ patno }),
        });

        if (!resp.ok) {
            document.getElementById('cif-info').textContent = 'Survival model not available';
            return;
        }

        survivalData = await resp.json();
        renderCIFChart(survivalData);

    } catch (err) {
        console.error('Survival panel error:', err);
        document.getElementById('cif-info').textContent = 'Error loading survival prediction';
    }
}

function renderCIFChart(data, counterfactualData = null) {
    if (!data) return;

    const traces = [];
    const timeBins = data.time_bin_months || TIME_BINS;

    // Get top 3 transitions for display
    const topTransitions = (data.top_transitions || []).slice(0, 3);
    const showCauses = topTransitions.map(t => t.cause_idx);

    if (showCauses.length === 0) {
        document.getElementById('cif-info').textContent = 'No significant transitions predicted';
        return;
    }

    showCauses.forEach((causeIdx, i) => {
        const stage = CAUSE_LABELS[causeIdx] || String(causeIdx);
        const color = STAGE_COLORS[stage] || '#7f8c8d';

        // DeepHit CIF (solid line)
        if (data.deephit_cif) {
            traces.push({
                x: timeBins,
                y: data.deephit_cif[causeIdx],
                mode: 'lines',
                name: `DeepHit → Stage ${stage}`,
                line: { color, width: 2, dash: 'solid' },
                legendgroup: `cause${causeIdx}`,
            });

            // Conformal band (shaded area)
            if (data.deephit_cif_bands) {
                const lower = data.deephit_cif_bands[causeIdx].map(b => b[0]);
                const upper = data.deephit_cif_bands[causeIdx].map(b => b[1]);

                traces.push({
                    x: [...timeBins, ...timeBins.slice().reverse()],
                    y: [...upper, ...lower.slice().reverse()],
                    fill: 'toself',
                    fillcolor: color + '15',
                    line: { width: 0 },
                    showlegend: false,
                    legendgroup: `cause${causeIdx}`,
                    hoverinfo: 'skip',
                });
            }
        }

        // Graph-DT CIF (dashed line)
        if (data.graph_dt_cif) {
            traces.push({
                x: timeBins,
                y: data.graph_dt_cif[causeIdx],
                mode: 'lines',
                name: `Graph-DT → Stage ${stage}`,
                line: { color, width: 2, dash: 'dash' },
                legendgroup: `cause${causeIdx}`,
            });
        }

        // Counterfactual overlay (dotted, if what-if is active)
        if (counterfactualData && counterfactualData.deephit_cif) {
            traces.push({
                x: timeBins,
                y: counterfactualData.deephit_cif[causeIdx],
                mode: 'lines',
                name: `What-If → Stage ${stage}`,
                line: { color, width: 2, dash: 'dot' },
                legendgroup: `cause${causeIdx}_cf`,
                opacity: 0.7,
            });
        }
    });

    const layout = {
        margin: { t: 10, r: 10, b: 50, l: 50 },
        xaxis: {
            title: { text: 'Months', font: { size: 11 } },
            type: 'log',
            tickvals: [3, 6, 12, 24, 60, 120, 180],
            ticktext: ['3', '6', '12', '24', '60', '120', '180'],
        },
        yaxis: {
            title: { text: 'Cumulative Incidence', font: { size: 11 } },
            range: [0, 1],
            tickformat: '.0%',
        },
        legend: {
            font: { size: 9 },
            orientation: 'h',
            y: -0.25,
        },
        showlegend: true,
        plot_bgcolor: '#fafbfc',
        paper_bgcolor: 'transparent',
        hovermode: 'x unified',
    };

    Plotly.newPlot('cif-chart', traces, layout, {
        responsive: true,
        displayModeBar: false,
    });

    // Info text with top transitions
    if (topTransitions.length > 0) {
        const infoHtml = topTransitions.map(t =>
            `<strong>Stage ${t.destination_stage}:</strong> ` +
            `${(t.cif_at_12mo * 100).toFixed(1)}% at 1yr, ` +
            `${(t.cif_at_36mo * 100).toFixed(1)}% at 3yr`
        ).join(' &nbsp;|&nbsp; ');
        document.getElementById('cif-info').innerHTML = infoHtml;
    }
}
