document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const reviewInput = document.getElementById('reviewInput');
    const charCounter = document.getElementById('charCounter');
    const clearBtn = document.getElementById('clearBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const sampleBtns = document.querySelectorAll('.sample-btn');
    
    // Tabs
    const tabSingle = document.getElementById('tabSingle');
    const tabBatch = document.getElementById('tabBatch');
    const singleInputView = document.getElementById('singleInputView');
    const batchInputView = document.getElementById('batchInputView');
    const batchInput = document.getElementById('batchInput');
    const batchAnalyzeBtn = document.getElementById('batchAnalyzeBtn');
    
    // Result Views
    const statusBadge = document.getElementById('statusBadge');
    const resultPlaceholder = document.getElementById('resultPlaceholder');
    const resultContent = document.getElementById('resultContent');
    const batchResultContent = document.getElementById('batchResultContent');
    const batchList = document.getElementById('batchList');
    
    // Result Details
    const sentimentBanner = document.getElementById('sentimentBanner');
    const sentimentIcon = document.getElementById('sentimentIcon');
    const sentimentValue = document.getElementById('sentimentValue');
    const confidenceValue = document.getElementById('confidenceValue');
    const analyzedTextQuote = document.getElementById('analyzedTextQuote');
    
    // Probabilities
    const probNegVal = document.getElementById('probNegVal');
    const probNegBar = document.getElementById('probNegBar');
    const probNeuVal = document.getElementById('probNeuVal');
    const probNeuBar = document.getElementById('probNeuBar');
    const probPosVal = document.getElementById('probPosVal');
    const probPosBar = document.getElementById('probPosBar');
    
    // History
    const historyList = document.getElementById('historyList');
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');
    
    // Modal
    const modelInfoBtn = document.getElementById('modelInfoBtn');
    const modelModal = document.getElementById('modelModal');
    const closeModalBtn = document.getElementById('closeModalBtn');
    const specSeqLen = document.getElementById('specSeqLen');
    const specClasses = document.getElementById('specClasses');
    const specVocab = document.getElementById('specVocab');

    let history = [];

    // --- Input Character Counter ---
    reviewInput.addEventListener('input', () => {
        const len = reviewInput.value.length;
        charCounter.textContent = `${len} character${len === 1 ? '' : 's'}`;
    });

    // --- Clear Button ---
    clearBtn.addEventListener('click', () => {
        reviewInput.value = '';
        charCounter.textContent = '0 characters';
        reviewInput.focus();
    });

    // --- Quick Sample Buttons ---
    sampleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const sampleText = btn.getAttribute('data-text');
            reviewInput.value = sampleText;
            charCounter.textContent = `${sampleText.length} characters`;
            // Trigger analysis directly
            analyzeSingleReview(sampleText);
        });
    });

    // --- Mode Switching Tabs ---
    tabSingle.addEventListener('click', () => {
        tabSingle.classList.add('active');
        tabBatch.classList.remove('active');
        singleInputView.classList.add('active');
        batchInputView.classList.remove('active');
        
        batchResultContent.classList.add('hidden');
        if (resultContent.classList.contains('active-result')) {
            resultContent.classList.remove('hidden');
            resultPlaceholder.classList.add('hidden');
        } else {
            resultPlaceholder.classList.remove('hidden');
        }
    });

    tabBatch.addEventListener('click', () => {
        tabBatch.classList.add('active');
        tabSingle.classList.remove('active');
        batchInputView.classList.add('active');
        singleInputView.classList.remove('active');
        
        resultContent.classList.add('hidden');
        resultPlaceholder.classList.add('hidden');
        batchResultContent.classList.remove('hidden');
    });

    batchInput.addEventListener('input', () => {
        const lines = batchInput.value.split('\n').filter(l => l.trim().length > 0);
        batchAnalyzeBtn.innerHTML = `<i class="fa-solid fa-layer-group"></i> Analyze Batch (${lines.length} item${lines.length === 1 ? '' : 's'})`;
    });

    // --- Analyze Single Review ---
    analyzeBtn.addEventListener('click', () => {
        const text = reviewInput.value.trim();
        if (!text) {
            alert('Please enter a review text first!');
            return;
        }
        analyzeSingleReview(text);
    });

    async function analyzeSingleReview(text) {
        setStatusAnalyzing(true);
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || 'Prediction failed');
            }

            const data = await response.json();
            displaySingleResult(text, data);
            addToHistory(text, data.sentiment, data.confidence);
        } catch (err) {
            alert('Error: ' + err.message);
        } finally {
            setStatusAnalyzing(false);
        }
    }

    function setStatusAnalyzing(isAnalyzing) {
        if (isAnalyzing) {
            statusBadge.className = 'status-indicator analyzing';
            statusBadge.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...';
            analyzeBtn.disabled = true;
        } else {
            statusBadge.className = 'status-indicator ready';
            statusBadge.innerHTML = '<i class="fa-solid fa-circle"></i> Model Ready';
            analyzeBtn.disabled = false;
        }
    }

    function displaySingleResult(text, data) {
        resultPlaceholder.classList.add('hidden');
        resultContent.classList.remove('hidden');
        resultContent.classList.add('active-result');

        const sentiment = data.sentiment.toLowerCase();
        
        // Banner style
        sentimentBanner.className = `sentiment-banner ${sentiment}`;
        sentimentValue.textContent = data.sentiment;
        confidenceValue.textContent = `${data.confidence}%`;
        analyzedTextQuote.textContent = `"${text}"`;

        // Icons
        if (sentiment === 'positive') {
            sentimentIcon.className = 'fa-solid fa-face-smile';
        } else if (sentiment === 'neutral') {
            sentimentIcon.className = 'fa-solid fa-face-meh';
        } else {
            sentimentIcon.className = 'fa-solid fa-face-frown';
        }

        // Probabilities
        const neg = data.breakdown['Negative'] || 0;
        const neu = data.breakdown['Neutral'] || 0;
        const pos = data.breakdown['Positive'] || 0;

        probNegVal.textContent = `${neg}%`;
        probNegBar.style.width = `${neg}%`;

        probNeuVal.textContent = `${neu}%`;
        probNeuBar.style.width = `${neu}%`;

        probPosVal.textContent = `${pos}%`;
        probPosBar.style.width = `${pos}%`;

        // Preprocessing Steps
        if (data.preprocessing) {
            const prepCleanedText = document.getElementById('prepCleanedText');
            const prepTokensList = document.getElementById('prepTokensList');
            const prepTokenCount = document.getElementById('prepTokenCount');

            if (prepCleanedText) prepCleanedText.textContent = data.preprocessing.cleaned_text;
            if (prepTokenCount) prepTokenCount.textContent = data.preprocessing.total_words_tokenized;

            if (prepTokensList && data.preprocessing.tokens_first_20) {
                const tokenLines = data.preprocessing.tokens_first_20.map((token, idx) => {
                    return `  <div class="token-row"><span class="token-idx">${idx}</span> : <span class="token-val">${token}</span></div>`;
                }).join('\n');
                prepTokensList.innerHTML = `[\n${tokenLines}\n]`;
            }
        }
    }

    // --- Batch Analysis ---
    batchAnalyzeBtn.addEventListener('click', async () => {
        const lines = batchInput.value.split('\n').map(l => l.trim()).filter(l => l.length > 0);
        if (lines.length === 0) {
            alert('Please enter at least one review text in batch mode.');
            return;
        }

        batchList.innerHTML = '<div class="empty-history"><i class="fa-solid fa-spinner fa-spin"></i> Processing batch reviews...</div>';
        
        const results = [];
        for (const line of lines) {
            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: line })
                });
                const resData = await response.json();
                results.push({ text: line, ...resData });
                addToHistory(line, resData.sentiment, resData.confidence);
            } catch (e) {
                results.push({ text: line, sentiment: 'Error', confidence: 0 });
            }
        }

        renderBatchResults(results);
    });

    function renderBatchResults(results) {
        batchList.innerHTML = '';
        results.forEach((item, index) => {
            const div = document.createElement('div');
            div.className = 'batch-item';
            const sentClass = (item.sentiment || 'neutral').toLowerCase();
            const icon = sentClass === 'positive' ? 'fa-face-smile' : sentClass === 'neutral' ? 'fa-face-meh' : 'fa-face-frown';
            
            div.innerHTML = `
                <div class="batch-item-text"><strong>#${index + 1}:</strong> "${escapeHtml(item.text)}"</div>
                <div class="history-badge ${sentClass}">
                    <i class="fa-solid ${icon}"></i> ${item.sentiment} (${item.confidence}%)
                </div>
            `;
            batchList.appendChild(div);
        });
    }

    // --- History Management ---
    function addToHistory(text, sentiment, confidence) {
        history.unshift({ text, sentiment, confidence, time: new Date().toLocaleTimeString() });
        renderHistory();
    }

    function renderHistory() {
        if (history.length === 0) {
            historyList.innerHTML = `
                <div class="empty-history">
                    <i class="fa-regular fa-folder-open"></i>
                    <p>No past analysis log yet.</p>
                </div>`;
            return;
        }

        historyList.innerHTML = '';
        history.forEach((item) => {
            const div = document.createElement('div');
            div.className = 'history-item';
            const sentClass = item.sentiment.toLowerCase();
            const icon = sentClass === 'positive' ? 'fa-face-smile' : sentClass === 'neutral' ? 'fa-face-meh' : 'fa-face-frown';

            div.innerHTML = `
                <div class="history-text" title="${escapeHtml(item.text)}">"${escapeHtml(item.text)}"</div>
                <div class="history-badge ${sentClass}">
                    <i class="fa-solid ${icon}"></i> ${item.sentiment} (${item.confidence}%)
                </div>
            `;
            historyList.appendChild(div);
        });
    }

    clearHistoryBtn.addEventListener('click', () => {
        history = [];
        renderHistory();
    });

    // Helper: Escape HTML
    function escapeHtml(str) {
        return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    }

    // --- Model Specs Modal ---
    modelInfoBtn.addEventListener('click', async () => {
        modelModal.classList.add('active');
        try {
            const response = await fetch('/api/model-info');
            const data = await response.json();
            specSeqLen.textContent = `${data.input_sequence_length} tokens`;
            specClasses.textContent = data.classes.join(', ');
            specVocab.textContent = `${data.vocabulary_size.toLocaleString()} words`;
        } catch (e) {
            specVocab.textContent = 'Error fetching model details';
        }
    });

    closeModalBtn.addEventListener('click', () => {
        modelModal.classList.remove('active');
    });

    modelModal.addEventListener('click', (e) => {
        if (e.target === modelModal) modelModal.classList.remove('active');
    });
});
