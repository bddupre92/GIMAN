/**
 * similarity_graph.js — D3.js force-directed patient similarity graph.
 */

async function loadSimilarityPanel(patno) {
    try {
        const resp = await fetch(`/api/patients/${patno}/neighbors`);
        if (!resp.ok) {
            document.getElementById('similarity-list').innerHTML =
                '<span class="text-clinical-400">Patient similarity not available for this patient</span>';
            return;
        }

        const neighbors = await resp.json();
        if (!neighbors || !neighbors.length) {
            document.getElementById('similarity-list').innerHTML =
                '<span class="text-clinical-400">No similar patients found in fold 0 graph</span>';
            return;
        }

        renderSimilarityGraph(patno, neighbors);
        renderSimilarityList(neighbors);

    } catch (err) {
        console.error('Similarity panel error:', err);
    }
}

function renderSimilarityGraph(targetPatno, neighbors) {
    const container = document.getElementById('similarity-graph');
    container.innerHTML = '';

    const width = container.clientWidth || 400;
    const height = container.clientHeight || 250;

    const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    // Build nodes and links
    const nodes = [
        {
            id: targetPatno,
            stage: patientData?.current_stage || '?',
            isTarget: true,
            visits: patientData?.n_visits || 0,
        },
    ];

    const links = [];

    neighbors.forEach(n => {
        nodes.push({
            id: n.patno,
            stage: n.current_stage,
            isTarget: false,
            visits: n.n_visits,
            similarity: n.similarity,
        });
        links.push({
            source: targetPatno,
            target: n.patno,
            strength: n.similarity,
        });
    });

    // Force simulation
    const simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id).distance(80).strength(d => d.strength * 0.5))
        .force('charge', d3.forceManyBody().strength(-120))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(20));

    // Links
    const link = svg.append('g')
        .selectAll('line')
        .data(links)
        .enter()
        .append('line')
        .attr('class', 'similarity-link')
        .attr('stroke-width', d => Math.max(1, d.strength * 3));

    // Nodes
    const node = svg.append('g')
        .selectAll('circle')
        .data(nodes)
        .enter()
        .append('circle')
        .attr('class', 'similarity-node')
        .attr('r', d => d.isTarget ? 14 : 9)
        .attr('fill', d => STAGE_COLORS[d.stage] || '#7f8c8d')
        .attr('stroke-width', d => d.isTarget ? 3 : 2)
        .call(d3.drag()
            .on('start', (event, d) => {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            })
            .on('drag', (event, d) => {
                d.fx = event.x;
                d.fy = event.y;
            })
            .on('end', (event, d) => {
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            })
        );

    // Labels
    const labels = svg.append('g')
        .selectAll('text')
        .data(nodes)
        .enter()
        .append('text')
        .attr('class', 'similarity-label')
        .attr('dy', d => d.isTarget ? -18 : -13)
        .text(d => d.isTarget ? `You (${d.id})` : d.id)
        .style('font-weight', d => d.isTarget ? 'bold' : 'normal');

    // Tooltips
    node.append('title')
        .text(d => `Patient ${d.id}\nStage: ${d.stage}\nVisits: ${d.visits}` +
            (d.similarity ? `\nSimilarity: ${(d.similarity * 100).toFixed(1)}%` : ''));

    // Node click → navigate to patient
    node.on('click', (event, d) => {
        if (!d.isTarget) {
            window.open(`/patient/${d.id}`, '_blank');
        }
    });

    // Tick
    simulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);
        node
            .attr('cx', d => Math.max(15, Math.min(width - 15, d.x)))
            .attr('cy', d => Math.max(15, Math.min(height - 15, d.y)));
        labels
            .attr('x', d => d.x)
            .attr('y', d => d.y);
    });
}

function renderSimilarityList(neighbors) {
    const container = document.getElementById('similarity-list');
    container.innerHTML = neighbors.slice(0, 5).map(n => {
        const color = STAGE_COLORS[n.current_stage] || '#7f8c8d';
        const trajectory = n.stage_trajectory?.join(' > ') || '--';
        return `
            <div class="flex items-center justify-between py-1 border-b border-clinical-50">
                <a href="/patient/${n.patno}" target="_blank" class="text-clinical-600 hover:text-clinical-800 font-medium">
                    ${n.patno}
                </a>
                <span class="stage-pill" style="background:${color}; font-size:10px;">${n.current_stage}</span>
                <span class="text-clinical-400 truncate max-w-[120px]" title="${trajectory}">${trajectory}</span>
                <span class="text-clinical-400">${(n.similarity * 100).toFixed(0)}%</span>
            </div>
        `;
    }).join('');
}
