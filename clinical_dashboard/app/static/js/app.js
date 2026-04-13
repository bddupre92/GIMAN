/**
 * app.js — Patient list page: search, filter, pagination, cohort overview.
 */

const STAGE_COLORS = {
    '0': '#2ecc71', '1': '#3498db', '2B': '#f1c40f',
    '3': '#e67e22', '4': '#e74c3c', '5': '#9b59b6', '6': '#7f8c8d',
};

let currentOffset = 0;
const PAGE_SIZE = 50;
let totalPatients = 0;

// ── Load cohort summary ────────────────────────────────────
async function loadCohortSummary() {
    try {
        const resp = await fetch('/api/cohort/summary');
        const data = await resp.json();

        document.getElementById('stat-patients').textContent = data.n_patients?.toLocaleString() ?? '--';
        document.getElementById('stat-visits').textContent = data.n_visits?.toLocaleString() ?? '--';
        document.getElementById('stat-transitions').textContent = data.n_transitions?.toLocaleString() ?? '--';

        // Stage distribution
        const dist = data.stage_distribution || {};
        const stages = Object.keys(dist).sort();
        document.getElementById('stat-stages').textContent = stages.length;

        const total = Object.values(dist).reduce((a, b) => a + b, 0);
        const bar = document.getElementById('stage-distribution-bar');
        const legend = document.getElementById('stage-distribution-legend');
        bar.innerHTML = '';
        legend.innerHTML = '';

        stages.forEach(stage => {
            const pct = (dist[stage] / total * 100);
            const color = STAGE_COLORS[stage] || '#7f8c8d';

            const seg = document.createElement('div');
            seg.style.width = pct + '%';
            seg.style.backgroundColor = color;
            seg.title = `Stage ${stage}: ${dist[stage]} (${pct.toFixed(1)}%)`;
            bar.appendChild(seg);

            const label = document.createElement('span');
            label.className = 'text-xs text-clinical-500';
            label.innerHTML = `<span style="color:${color}">&#9679;</span> ${stage}: ${dist[stage]}`;
            legend.appendChild(label);
        });

        document.getElementById('cohort-data-label').textContent =
            `${data.n_patients} patients, ${data.n_visits} visits`;
    } catch (err) {
        console.error('Failed to load cohort summary:', err);
    }
}

// ── Load patient list ──────────────────────────────────────
async function loadPatients() {
    const search = document.getElementById('patient-search').value.trim();
    const stage = document.getElementById('stage-filter').value;

    let url = `/api/patients?limit=${PAGE_SIZE}&offset=${currentOffset}`;
    if (search) url += `&search=${encodeURIComponent(search)}`;
    if (stage) url += `&stage=${encodeURIComponent(stage)}`;

    try {
        const resp = await fetch(url);
        const data = await resp.json();

        totalPatients = data.total;
        renderPatientTable(data.patients);
        updatePagination();

        document.getElementById('patient-count').textContent =
            `${data.total} patients found`;
    } catch (err) {
        console.error('Failed to load patients:', err);
        document.getElementById('patient-table-body').innerHTML =
            '<tr><td colspan="9" class="px-4 py-8 text-center text-red-500">Failed to load patient data</td></tr>';
    }
}

function renderPatientTable(patients) {
    const tbody = document.getElementById('patient-table-body');

    if (!patients.length) {
        tbody.innerHTML =
            '<tr><td colspan="9" class="px-4 py-8 text-center text-clinical-400">No patients found</td></tr>';
        return;
    }

    tbody.innerHTML = patients.map(p => {
        const color = STAGE_COLORS[p.current_stage] || '#7f8c8d';
        const initColor = STAGE_COLORS[p.initial_stage] || '#7f8c8d';
        return `
            <tr class="patient-row" onclick="window.location='/patient/${p.patno}'">
                <td class="px-4 py-3 font-medium text-clinical-700">${p.patno}</td>
                <td class="px-4 py-3">${p.age_at_baseline?.toFixed(1) ?? '--'}</td>
                <td class="px-4 py-3">${p.sex === 1 ? 'F' : 'M'}</td>
                <td class="px-4 py-3 text-center">
                    <span class="stage-pill" style="background:${color}">
                        ${p.current_stage}
                    </span>
                </td>
                <td class="px-4 py-3 text-center">
                    <span class="stage-pill" style="background:${initColor}">
                        ${p.initial_stage}
                    </span>
                </td>
                <td class="px-4 py-3 text-right">${p.n_visits}</td>
                <td class="px-4 py-3 text-right">${p.follow_up_months?.toFixed(0) ?? '--'}</td>
                <td class="px-4 py-3 text-center">
                    ${p.has_transitions ?
                        '<span class="text-green-600 font-medium">Yes</span>' :
                        '<span class="text-clinical-300">No</span>'}
                </td>
                <td class="px-4 py-3">
                    <a href="/patient/${p.patno}" class="text-clinical-500 hover:text-clinical-700">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/>
                        </svg>
                    </a>
                </td>
            </tr>
        `;
    }).join('');
}

function updatePagination() {
    const totalPages = Math.ceil(totalPatients / PAGE_SIZE);
    const currentPage = Math.floor(currentOffset / PAGE_SIZE) + 1;

    document.getElementById('page-info').textContent = `Page ${currentPage} of ${totalPages}`;
    document.getElementById('prev-page').disabled = currentOffset === 0;
    document.getElementById('next-page').disabled = currentOffset + PAGE_SIZE >= totalPatients;
}

// ── Event listeners ────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    loadCohortSummary();
    loadPatients();

    // Search debounce
    let searchTimeout;
    document.getElementById('patient-search').addEventListener('input', () => {
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => {
            currentOffset = 0;
            loadPatients();
        }, 300);
    });

    // Stage filter
    document.getElementById('stage-filter').addEventListener('change', () => {
        currentOffset = 0;
        loadPatients();
    });

    // Pagination
    document.getElementById('prev-page').addEventListener('click', () => {
        currentOffset = Math.max(0, currentOffset - PAGE_SIZE);
        loadPatients();
    });

    document.getElementById('next-page').addEventListener('click', () => {
        currentOffset += PAGE_SIZE;
        loadPatients();
    });
});
