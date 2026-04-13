/**
 * trajectory_chart.js — Plotly.js stage trajectory timeline.
 */

function renderTrajectoryChart(data) {
    if (!data || !data.visit_times_months || !data.stage_trajectory) return;

    const times = data.visit_times_months;
    const stages = data.stage_trajectory;

    // Map stages to numeric Y values for plotting
    const stageOrder = { '0': 0, '1': 1, '2B': 2, '3': 3, '4': 4, '5': 5, '6': 6 };
    const yVals = stages.map(s => stageOrder[s] ?? 3);
    const colors = stages.map(s => STAGE_COLORS[s] || '#7f8c8d');

    // Detect transitions (color segments red=forward, green=backward)
    const segmentColors = [];
    for (let i = 1; i < stages.length; i++) {
        const prev = stageOrder[stages[i - 1]] ?? 3;
        const curr = stageOrder[stages[i]] ?? 3;
        if (curr > prev) segmentColors.push('#e74c3c');      // Forward = red
        else if (curr < prev) segmentColors.push('#2ecc71');  // Backward = green
        else segmentColors.push('#bcccdc');                    // Same = gray
    }

    // Main scatter (visits as dots)
    const trace = {
        x: times,
        y: yVals,
        mode: 'lines+markers',
        type: 'scatter',
        marker: {
            size: 8,
            color: colors,
            line: { width: 1, color: '#fff' },
        },
        line: { color: '#bcccdc', width: 1 },
        text: stages.map((s, i) => `Visit ${i + 1}: Stage ${s} at ${times[i]}mo`),
        hoverinfo: 'text',
    };

    const layout = {
        margin: { t: 10, r: 20, b: 40, l: 50 },
        xaxis: {
            title: { text: 'Months from Baseline', font: { size: 11 } },
            zeroline: false,
        },
        yaxis: {
            title: { text: 'NSD-ISS Stage', font: { size: 11 } },
            tickvals: [0, 1, 2, 3, 4, 5, 6],
            ticktext: ['0', '1', '2B', '3', '4', '5', '6'],
            range: [-0.5, 6.5],
        },
        showlegend: false,
        plot_bgcolor: '#fafbfc',
        paper_bgcolor: 'transparent',
    };

    Plotly.newPlot('trajectory-chart', [trace], layout, {
        responsive: true,
        displayModeBar: false,
    });
}
