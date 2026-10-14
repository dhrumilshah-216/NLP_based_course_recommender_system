console.log("script.js loaded successfully");

const reviewRequests = new Map();
let courseCatalogueRequest;

function switchTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));

    document.getElementById('content-' + tabName).classList.add('active');
    document.getElementById('btn-' + tabName).classList.add('active');
}

function fillSearch(text) {
    document.getElementById('searchInput').value = text;
    performSearch();
}

function escapeHTML(value) {
    return String(value ?? '').replace(/[&<>"']/g, character => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
    })[character]);
}

function setButtonLoading(button, loading) {
    button.disabled = loading;
    button.querySelector('.btn-text').style.display = loading ? 'none' : 'inline';
    button.querySelector('.btn-loader').style.display = loading ? 'inline' : 'none';
}

function showResultsState(message, isError = false) {
    const section = document.getElementById('resultsSection');
    const container = document.getElementById('resultsContainer');
    section.style.display = 'block';
    container.replaceChildren();
    const state = document.createElement('p');
    state.className = `state-message${isError ? ' error' : ''}`;
    state.setAttribute('role', isError ? 'alert' : 'status');
    state.textContent = message;
    container.appendChild(state);
}

async function loadCourseCatalogue() {
    if (!courseCatalogueRequest) {
        courseCatalogueRequest = fetch('/api/courses')
            .then(async response => {
                const data = await response.json();
                if (!response.ok) throw new Error(data.error || data.detail || 'Course details are unavailable');
                return new Map((data.courses || []).map(course => [String(course.course_id), course]));
            })
            .catch(error => {
                courseCatalogueRequest = null;
                throw error;
            });
    }
    return courseCatalogueRequest;
}

function formatMetadata(value) {
    if (Array.isArray(value)) return value.filter(Boolean).join(', ');
    return String(value ?? '').trim();
}

function sentimentCounts(counts) {
    const labels = ['positive', 'negative', 'neutral', 'mixed'];
    return labels.map(label => `${label}: ${Number(counts?.[label] || 0)}`).join(' · ');
}

function appendReviewAspect(container, aspect, summary) {
    const row = document.createElement('div');
    row.className = 'review-aspect';
    const heading = document.createElement('strong');
    heading.textContent = aspect.replaceAll('_', ' ');
    row.appendChild(heading);

    const prediction = document.createElement('p');
    prediction.textContent = `Rule-based prediction (${summary.prediction?.review_count || 0} assessed): ${sentimentCounts(summary.prediction?.sentiment_counts)}`;
    row.appendChild(prediction);

    const reference = summary.synthetic_reference_labels || {};
    const labels = document.createElement('p');
    labels.textContent = `Synthetic reference labels (${reference.labeled_review_count || 0}; not model predictions): ${sentimentCounts(reference.sentiment_counts)}`;
    row.appendChild(labels);
    container.appendChild(row);
}

function renderReviewPanel(panel, analysis) {
    panel.replaceChildren();
    const notice = document.createElement('p');
    notice.className = 'review-notice';
    notice.textContent = analysis.data_provenance?.notice || 'Synthetic sample review analysis; not actual student feedback.';
    panel.appendChild(notice);

    const overall = document.createElement('p');
    overall.className = 'review-method';
    overall.textContent = `Reviews: ${analysis.review_count || 0}. Overall sentiment is a separate synthetic reference label (not a model prediction): ${sentimentCounts(analysis.overall_sentiment_reference?.sentiment_counts)}.`;
    panel.appendChild(overall);

    const method = document.createElement('p');
    method.className = 'review-method';
    method.textContent = analysis.analysis_method?.aspect_prediction || 'Aspect predictions use keyword rules and are not a trained model.';
    panel.appendChild(method);

    const aspects = analysis.aspects || {};
    if (!Object.keys(aspects).length) {
        const empty = document.createElement('p');
        empty.className = 'state-message';
        empty.textContent = 'No aspect-level review information is available for this course.';
        panel.appendChild(empty);
        return;
    }
    for (const [aspect, summary] of Object.entries(aspects)) {
        appendReviewAspect(panel, aspect, summary);
    }
}

async function loadReviewAnalysis(courseId, panel, retry = false) {
    if (retry) reviewRequests.delete(courseId);
    panel.replaceChildren();
    const loading = document.createElement('p');
    loading.className = 'state-message';
    loading.textContent = 'Loading synthetic review insights...';
    panel.appendChild(loading);

    try {
        if (!reviewRequests.has(courseId)) {
            reviewRequests.set(courseId, fetch(`/api/reviews/${encodeURIComponent(courseId)}/analysis`)
                .then(async response => {
                    const data = await response.json();
                    if (!response.ok) throw new Error(data.detail || data.error || 'Review analysis is unavailable');
                    return data;
                }));
        }
        const analysis = await reviewRequests.get(courseId);
        renderReviewPanel(panel, analysis);
    } catch (error) {
        reviewRequests.delete(courseId);
        panel.replaceChildren();
        const message = document.createElement('p');
        message.className = 'state-message error';
        message.setAttribute('role', 'alert');
        message.textContent = error.message || 'Review analysis could not be loaded.';
        panel.appendChild(message);
        const retryButton = document.createElement('button');
        retryButton.type = 'button';
        retryButton.className = 'review-retry';
        retryButton.textContent = 'Try again';
        retryButton.addEventListener('click', () => loadReviewAnalysis(courseId, panel, true));
        panel.appendChild(retryButton);
    }
}

function resultCard(result, index, mode, course) {
    const courseId = result.course_id || course?.course_id || '';
    const title = result.title || course?.title || course?.course_title || 'Course title unavailable';
    const description = result.description || course?.description || '';
    let scoreLabel = '';

    if (mode === 'recommendation') {
        scoreLabel = Number.isFinite(Number(result.recommendation_score))
            ? `Recommendation ranking score: ${Number(result.recommendation_score).toFixed(3)} (not a probability)`
            : 'Recommendation score unavailable';
    } else {
        scoreLabel = Number.isFinite(Number(result.relevance_score))
            ? `Cosine relevance: ${Number(result.relevance_score).toFixed(3)} (not a probability)`
            : 'Relevance score unavailable';
    }

    const metadataFields = [
        ['Department', course?.department],
        ['Topics', course?.syllabus_topics],
        ['Prerequisites', course?.prerequisites],
        ['Tools', course?.tools_technologies]
    ];
    const metadata = metadataFields
        .map(([label, value]) => [label, formatMetadata(value)])
        .filter(([, value]) => value && !['none', 'n/a', 'na'].includes(value.toLowerCase()))
        .map(([label, value]) => `<span><strong>${escapeHTML(label)}:</strong> ${escapeHTML(value)}</span>`)
        .join(' · ');

    const explanation = mode === 'recommendation'
        ? result.why_recommended
        : result.match_explanation;
    const tradeoffs = result.tradeoffs;
    const reasonsHTML = Array.isArray(explanation) && explanation.length
        ? `<ul class="recommendation-reasons">${explanation.map(reason => `<li>${escapeHTML(reason)}</li>`).join('')}</ul>`
        : '';
    const tradeoffsHTML = Array.isArray(tradeoffs) && tradeoffs.length
        ? `<ul class="recommendation-tradeoffs">${tradeoffs.map(reason => `<li>${escapeHTML(reason)}</li>`).join('')}</ul>`
        : '';
    const componentsHTML = mode === 'recommendation' && result.components
        ? `<p class="course-metadata">Ranking contributions: ${Object.entries(result.components).map(([key, value]) =>
            `${escapeHTML(key.replaceAll('_', ' '))} ${Number(value.score).toFixed(2)} × ${Number(value.weight).toFixed(2)}`
        ).join(' · ')}</p>`
        : '';
    const recommendationSignals = mode === 'recommendation'
        ? `<p class="course-metadata">Semantic relevance (cosine): ${result.semantic_relevance == null ? 'not available' : Number(result.semantic_relevance).toFixed(3)} · Preference match: ${result.preference_match == null ? 'not supplied' : Number(result.preference_match).toFixed(3)}. These are ranking signals, not probabilities.</p>`
        : '';

    return `
        <article class="course-card" style="animation-delay: ${index * 0.1}s">
            <div class="course-header">
                <h3 class="course-title">${escapeHTML(title)}</h3>
                <div class="similarity-badge">${escapeHTML(scoreLabel)}</div>
            </div>
            <p class="course-id">${escapeHTML(courseId || 'Course ID unavailable')}</p>
            <p class="course-description">${escapeHTML(description || 'Course description is not available.')}</p>
            ${metadata ? `<p class="course-metadata">${metadata}</p>` : ''}
            ${reasonsHTML}
            ${tradeoffsHTML}
            ${recommendationSignals}
            ${componentsHTML}
            ${courseId ? `
                <details class="review-details" data-course-id="${escapeHTML(courseId)}">
                    <summary>Review insights (synthetic sample reviews)</summary>
                    <div class="review-panel"></div>
                </details>` : ''}
        </article>`;
}

async function renderCourseResults(mode, data) {
    const section = document.getElementById('resultsSection');
    const container = document.getElementById('resultsContainer');
    const results = mode === 'recommendation' ? data.recommendations : data.results;
    const heading = mode === 'recommendation' ? 'Personalized recommendations' : 'Semantic search results';
    document.getElementById('resultsHeading').textContent = heading;
    document.getElementById('resultsInfo').textContent = `${data.total_found || 0} course${data.total_found === 1 ? '' : 's'} returned`;
    section.style.display = 'block';

    if (!Array.isArray(results) || results.length === 0) {
        showResultsState('No courses matched this request. Try a broader query or adjust your preferences.');
        return;
    }

    let coursesById = new Map();
    try {
        coursesById = await loadCourseCatalogue();
    } catch (error) {
        document.getElementById('resultsInfo').textContent += ' · Some course details may be unavailable';
    }
    container.innerHTML = results.map((result, index) => resultCard(
        result, index, mode, coursesById.get(String(result.course_id))
    )).join('');

    container.querySelectorAll('.review-details').forEach(details => {
        details.addEventListener('toggle', () => {
            if (!details.open || details.dataset.loaded === 'true') return;
            details.dataset.loaded = 'true';
            loadReviewAnalysis(details.dataset.courseId, details.querySelector('.review-panel'));
        });
    });
}

async function performSearch() {
    const query = document.getElementById('searchInput').value.trim();
    if (!query) {
        showResultsState('Enter a learning goal or topic to search for courses.', true);
        return;
    }

    const button = document.getElementById('searchBtn');
    setButtonLoading(button, true);
    document.getElementById('resultsHeading').textContent = 'Semantic search results';
    document.getElementById('resultsInfo').textContent = 'Searching the course catalogue...';
    showResultsState('Finding courses that match your query...');

    try {
        const res = await fetch('/api/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, top_k: 5 })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || data.detail || 'Search failed');
        await renderCourseResults('search', data);
    } catch (error) {
        document.getElementById('resultsInfo').textContent = 'Search could not be completed';
        showResultsState(error.message || 'Search failed. Please try again.', true);
    } finally {
        setButtonLoading(button, false);
    }
}

function readPreferences() {
    const preferenceFields = {
        interests: 'prefInterests',
        career_goal: 'prefCareerGoal',
        difficulty: 'prefDifficulty',
        workload: 'prefWorkload',
        learning_style: 'prefLearningStyle',
        math_complexity: 'prefMathComplexity',
        scoring_preference: 'prefScoring'
    };
    const preferences = {};
    for (const [key, id] of Object.entries(preferenceFields)) {
        const value = document.getElementById(id).value.trim();
        if (value) preferences[key] = value;
    }
    const completedCourses = document.getElementById('completedCourses').value
        .split(',')
        .map(courseId => courseId.trim())
        .filter(Boolean);
    return { preferences, completedCourses };
}

async function performRecommendations() {
    const query = document.getElementById('searchInput').value.trim();
    const { preferences, completedCourses } = readPreferences();
    if (!query && Object.keys(preferences).length === 0) {
        showResultsState('Enter a query or choose at least one preference to get recommendations.', true);
        return;
    }

    const button = document.getElementById('recommendBtn');
    setButtonLoading(button, true);
    document.getElementById('resultsHeading').textContent = 'Personalized recommendations';
    document.getElementById('resultsInfo').textContent = 'Ranking courses against your query and preferences...';
    showResultsState('Preparing personalized recommendations...');

    try {
        const res = await fetch('/api/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query || null,
                top_k: 5,
                preferences,
                completed_courses: completedCourses.length ? completedCourses : null,
                enforce_prerequisites: document.getElementById('enforcePrerequisites').checked
            })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || data.detail || 'Recommendations failed');
        await renderCourseResults('recommendation', data);
        if (data.data_notice) {
            document.getElementById('resultsInfo').textContent += ' · Demonstration data; ranking is not real-world validation';
        }
    } catch (error) {
        document.getElementById('resultsInfo').textContent = 'Recommendations could not be completed';
        showResultsState(error.message || 'Recommendations failed. Please try again.', true);
    } finally {
        setButtonLoading(button, false);
    }
}

async function performClassicalAnalysis() {
    const text = document.getElementById('classicalInput').value.trim();
    const button = document.getElementById('classicalBtn');
    const section = document.getElementById('classicalResults');
    const container = document.getElementById('classicalContainer');
    section.style.display = 'block';

    if (!text) {
        container.textContent = 'Enter text to analyze.';
        return;
    }

    setButtonLoading(button, true);
    container.textContent = 'Processing...';
    try {
        const res = await fetch('/api/classical-analysis', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || data.message || data.detail || 'Analysis failed');

        container.replaceChildren();
        const analysis = data.analysis || Object.fromEntries(
            Object.entries(data).filter(([key]) => /^\d+_/.test(key))
        );
        if (!Object.keys(analysis).length) {
            container.textContent = 'The analysis service returned no result fields.';
            return;
        }
        for (const [key, value] of Object.entries(analysis)) {
            const card = document.createElement('div');
            card.className = 'analysis-card';
            const heading = document.createElement('h4');
            heading.textContent = key.replace(/_/g, ' ').replace(/\b\w/g, letter => letter.toUpperCase());
            const output = document.createElement('pre');
            output.textContent = JSON.stringify(value, null, 2);
            card.append(heading, output);
            container.appendChild(card);
        }
    } catch (error) {
        container.replaceChildren();
        const message = document.createElement('p');
        message.className = 'state-message error';
        message.setAttribute('role', 'alert');
        message.textContent = error.message || 'Analysis failed. Please try again.';
        container.appendChild(message);
    } finally {
        setButtonLoading(button, false);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('searchInput').addEventListener('keydown', event => {
        if (event.key === 'Enter') performSearch();
    });
    document.getElementById('classicalInput').addEventListener('keydown', event => {
        if (event.key === 'Enter') performClassicalAnalysis();
    });
});
