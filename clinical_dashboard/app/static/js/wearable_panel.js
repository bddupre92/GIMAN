/**
 * wearable_panel.js — Synthetic wearable data display with SYNTHETIC banner.
 */

async function loadWearablePanel(patno) {
    try {
        const resp = await fetch(`/api/wearable/${patno}`);

        if (!resp.ok) {
            document.getElementById('wearable-info').textContent =
                'No synthetic wearable data available for this patient.';
            return;
        }

        const data = await resp.json();
        renderWearableChart(data);

    } catch (err) {
        console.error('Wearable panel error:', err);
        document.getElementById('wearable-info').textContent =
            'Wearable data module not yet configured.';
    }
}

function renderWearableChart(data) {
    if (!data || !data.sensor_data) {
        document.getElementById('wearable-info').textContent =
            'Wearable data not available for this patient.';
        return;
    }

    const sensors = data.sensor_data;
    const traces = [];

    // Gait speed
    if (sensors.gait_speed) {
        traces.push({
            x: sensors.gait_speed.dates || sensors.gait_speed.map((_, i) => i),
            y: sensors.gait_speed.values || sensors.gait_speed,
            name: 'Gait Speed (m/s)',
            type: 'scatter',
            mode: 'lines',
            line: { color: '#3498db', width: 1.5 },
            yaxis: 'y',
        });
    }

    // Tremor amplitude
    if (sensors.tremor_amplitude) {
        traces.push({
            x: sensors.tremor_amplitude.dates || sensors.tremor_amplitude.map((_, i) => i),
            y: sensors.tremor_amplitude.values || sensors.tremor_amplitude,
            name: 'Tremor (g)',
            type: 'scatter',
            mode: 'lines',
            line: { color: '#e74c3c', width: 1.5 },
            yaxis: 'y2',
        });
    }

    // Sleep hours
    if (sensors.sleep_hours) {
        traces.push({
            x: sensors.sleep_hours.dates || sensors.sleep_hours.map((_, i) => i),
            y: sensors.sleep_hours.values || sensors.sleep_hours,
            name: 'Sleep (hrs)',
            type: 'scatter',
            mode: 'lines',
            line: { color: '#9b59b6', width: 1.5 },
            yaxis: 'y3',
        });
    }

    if (traces.length === 0) {
        document.getElementById('wearable-info').textContent =
            'No sensor data streams available.';
        return;
    }

    const layout = {
        margin: { t: 10, r: 60, b: 30, l: 50 },
        xaxis: { title: { text: 'Day', font: { size: 10 } } },
        yaxis: {
            title: { text: 'Gait (m/s)', font: { size: 9, color: '#3498db' } },
            side: 'left',
        },
        yaxis2: {
            title: { text: 'Tremor (g)', font: { size: 9, color: '#e74c3c' } },
            overlaying: 'y',
            side: 'right',
        },
        legend: {
            font: { size: 9 },
            orientation: 'h',
            y: -0.2,
        },
        showlegend: true,
        plot_bgcolor: '#fffaf0',
        paper_bgcolor: 'transparent',
    };

    Plotly.newPlot('wearable-chart', traces, layout, {
        responsive: true,
        displayModeBar: false,
    });

    document.getElementById('wearable-info').textContent =
        `Source: ${data.source} | ${data.disclaimer}`;
}
