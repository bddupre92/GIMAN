/**
 * staging_panel.js — CatBoost staging probabilities + feature importance.
 */

async function loadStagingPanel(patno) {
    try {
        const resp = await fetch('/api/staging/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ patno }),
        });

        if (!resp.ok) {
            document.getElementById('staging-info').textContent = 'Staging model not available';
            return;
        }

        stagingData = await resp.json();
        renderStagingChart(stagingData);
        renderImportanceChart(stagingData);

    } catch (err) {
        console.error('Staging panel error:', err);
        document.getElementById('staging-info').textContent = 'Error loading staging prediction';
    }
}

function renderStagingChart(data) {
    if (!data || !data.probabilities) return;

    const stages = Object.keys(data.probabilities);
    const probs = Object.values(data.probabilities);
    const colors = stages.map(s => STAGE_COLORS[s] || '#7f8c8d');

    const trace = {
        x: stages.map(s => `Stage ${s}`),
        y: probs,
        type: 'bar',
        marker: {
            color: colors,
            line: { width: 1, color: '#fff' },
        },
        text: probs.map(p => `${(p * 100).toFixed(1)}%`),
        textposition: 'outside',
        textfont: { size: 10 },
        hoverinfo: 'x+y',
    };

    const layout = {
        margin: { t: 5, r: 10, b: 30, l: 40 },
        yaxis: {
            title: { text: 'Probability', font: { size: 10 } },
            range: [0, Math.max(...probs) * 1.3],
            tickformat: '.0%',
        },
        xaxis: { tickfont: { size: 10 } },
        showlegend: false,
        plot_bgcolor: '#fafbfc',
        paper_bgcolor: 'transparent',
    };

    Plotly.newPlot('staging-chart', [trace], layout, {
        responsive: true,
        displayModeBar: false,
    });

    // Info text
    const conformalSet = data.conformal_set || [];
    document.getElementById('staging-info').innerHTML =
        `<strong>Predicted:</strong> Stage ${data.predicted_stage} ` +
        `<span class="text-clinical-400">(confidence: ${(data.confidence_level * 100).toFixed(0)}%)</span><br>` +
        `<strong>Conformal set:</strong> {${conformalSet.join(', ')}}`;
}

function renderImportanceChart(data) {
    if (!data || !data.features_used) return;

    // Load feature importance from explainability endpoint
    fetch(`/api/explain/feature_importance/${PATNO}`)
    .then(r => r.ok ? r.json() : null).then(explainData => {
        if (!explainData || !explainData.feature_importances) return;

        const top8 = explainData.feature_importances.slice(0, 8);
        const labels = top8.map(f => f.feature.replace('_', ' '));
        const values = top8.map(f => f.importance);

        const trace = {
            y: labels.reverse(),
            x: values.reverse(),
            type: 'bar',
            orientation: 'h',
            marker: { color: '#486581' },
            text: values.map(v => `${(v * 100).toFixed(1)}%`),
            textposition: 'outside',
            textfont: { size: 9 },
        };

        const layout = {
            margin: { t: 5, r: 50, b: 20, l: 110 },
            xaxis: { title: { text: 'Importance', font: { size: 10 } }, tickformat: '.0%' },
            yaxis: { tickfont: { size: 9 } },
            showlegend: false,
            plot_bgcolor: '#fafbfc',
            paper_bgcolor: 'transparent',
        };

        Plotly.newPlot('importance-chart', [trace], layout, {
            responsive: true,
            displayModeBar: false,
        });
    }).catch(() => {});
}
