/**
 * annotations_panel.js — Clinician notes CRUD interface.
 */

async function loadAnnotationsPanel(patno) {
    try {
        const resp = await fetch(`/api/annotations/${patno}`);
        if (!resp.ok) return;

        const data = await resp.json();
        renderAnnotations(data.annotations || []);
    } catch (err) {
        console.error('Annotations error:', err);
    }
}

function renderAnnotations(annotations) {
    const container = document.getElementById('annotations-list');

    if (!annotations.length) {
        container.innerHTML =
            '<div class="text-xs text-clinical-300 italic py-2">No annotations yet. Add the first clinical note below.</div>';
        return;
    }

    container.innerHTML = annotations.map(a => {
        const categoryClass = a.category.toLowerCase().replace(/\s+/g, '-');
        const ts = a.timestamp ? new Date(a.timestamp).toLocaleString() : '';
        return `
            <div class="annotation-card ${categoryClass}">
                <div class="flex justify-between items-start mb-1">
                    <span class="text-xs font-medium text-clinical-600">${a.category}</span>
                    <span class="text-xs text-clinical-400">${ts}</span>
                </div>
                <p class="text-xs text-clinical-700">${escapeHtml(a.text)}</p>
                <div class="text-xs text-clinical-400 mt-1">— ${escapeHtml(a.author)}</div>
            </div>
        `;
    }).join('');
}

async function addAnnotation() {
    const text = document.getElementById('annotation-text').value.trim();
    if (!text) return;

    const category = document.getElementById('annotation-category').value;

    try {
        const resp = await fetch(`/api/annotations/${PATNO}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text,
                category,
                author: 'Clinician',
            }),
        });

        if (!resp.ok) {
            alert('Failed to save annotation');
            return;
        }

        const data = await resp.json();
        renderAnnotations(data.annotations || []);

        // Clear input
        document.getElementById('annotation-text').value = '';

    } catch (err) {
        console.error('Add annotation error:', err);
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
