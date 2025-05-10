// --- Tab Navigation ---
document.addEventListener('DOMContentLoaded', function() {
    // Tab elements
    const singleTabBtn = document.getElementById('singleTabBtn');
    const batchTabBtn = document.getElementById('batchTabBtn');
    const historyTabBtn = document.getElementById('historyTabBtn');
    const singleSection = document.getElementById('singleSection');
    const batchSection = document.getElementById('batchSection');
    const historySection = document.getElementById('historySection');

    // Function to switch tabs
    function switchTab(tab) {
        // Remove active class from all tabs and sections
        [singleTabBtn, batchTabBtn, historyTabBtn].forEach(btn => btn.classList.remove('active'));
        [singleSection, batchSection, historySection].forEach(section => section.classList.remove('active'));
        
        // Add active class to selected tab and section
        tab.classList.add('active');
        
        if (tab === singleTabBtn) {
            singleSection.classList.add('active');
        } else if (tab === batchTabBtn) {
            batchSection.classList.add('active');
        } else if (tab === historyTabBtn) {
            historySection.classList.add('active');
            sessionHistory.updateDisplay();
        }
    }

    // Add event listeners to tab buttons
    singleTabBtn.addEventListener('click', () => switchTab(singleTabBtn));
    batchTabBtn.addEventListener('click', () => switchTab(batchTabBtn));
    historyTabBtn.addEventListener('click', () => switchTab(historyTabBtn));

    // --- Session History Feature ---
    const sessionHistory = {
        items: [],
        maxItems: 10, // Keep last 10 analyses

        add: function(analysisData) {
            analysisData.timestamp = new Date().toISOString();
            this.items.unshift(analysisData);
            if (this.items.length > this.maxItems) {
                this.items = this.items.slice(0, this.maxItems);
            }
            this.updateDisplay();
        },

        updateDisplay: function() {
            const historyContainer = document.getElementById('historyContainer');
            if (!historyContainer) return;
            
            if (this.items.length === 0) {
                historyContainer.innerHTML = `
                    <div class="empty-state">
                        <i class="fas fa-inbox"></i>
                        <h3>No Analysis History Yet</h3>
                        <p>Your analyzed images will appear here for future reference</p>
                    </div>
                `;
                return;
            }

            historyContainer.innerHTML = this.items.map((item, index) => `
                <div class="history-item">
                    <div class="history-images">
                        <img src="/uploads/${item.original_image}" alt="Original">
                        <img src="/uploads/${item.heatmap_image}" alt="Analysis">
                    </div>
                    <div class="history-details">
                        <h4>Analysis ${index + 1}</h4>
                        <p><strong>Type:</strong> ${item.fault_type}</p>
                        <p><strong>Confidence:</strong> ${item.confidence}%</p>
                        <div class="history-actions">
                            <button class="btn secondary-btn history-btn" onclick="sessionHistory.load(${index})">
                                <i class="fas fa-eye"></i> View
                            </button>
                            <button class="btn danger-btn history-btn delete-btn" onclick="sessionHistory.remove(${index})">
                                <i class="fas fa-trash"></i> Delete
                            </button>
                        </div>
                    </div>
                </div>
            `).join('');
        },

        load: function(index, pushToHistory = true) {
            if (index >= 0 && index < this.items.length) {
                const item = this.items[index];
                displaySingleResult(item, pushToHistory);
                window.scrollTo({ top: 0, behavior: 'smooth' });
                
                // Switch to single analysis tab
                switchTab(singleTabBtn);
                
                // Push state if requested (for direct View from history)
                if (pushToHistory) {
                    history.pushState(
                        { type: 'single', data: item },
                        '',
                        `?analysis=${encodeURIComponent(item.original_image)}`
                    );
                }
            }
        },

        remove: function(index) {
            if (index >= 0 && index < this.items.length) {
                if (confirm('Are you sure you want to delete this analysis from history?')) {
                    this.items.splice(index, 1);
                    this.updateDisplay();
                }
            }
        },

        clear: function() {
            if (this.items.length === 0) return;
            if (confirm('Are you sure you want to clear all analysis history?')) {
                this.items = [];
                this.updateDisplay();
            }
        }
    };

    // --- Main Script ---
    // Single image upload elements
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const dropArea = document.getElementById('dropArea');
    const fileInfo = document.getElementById('fileInfo');
    const resultsSection = document.getElementById('resultsSection');
    const originalImage = document.getElementById('originalImage');
    const heatmapImage = document.getElementById('heatmapImage');
    const faultType = document.getElementById('faultType');
    const confidence = document.getElementById('confidence');
    const recommendation = document.getElementById('recommendation');
    const statusBadge = document.getElementById('statusBadge');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const analyzeAnotherBtn = document.getElementById('analyzeAnotherBtn');
    const downloadReportBtn = document.getElementById('downloadReportBtn');

    // Batch upload elements
    const batchFileInput = document.getElementById('batchFileInput');
    const batchUploadBtn = document.getElementById('batchUploadBtn');
    const batchDropArea = document.getElementById('batchDropArea');
    const batchFileInfo = document.getElementById('batchFileInfo');
    const batchResultsSection = document.getElementById('batchResultsSection');
    const resultsGrid = document.getElementById('resultsGrid');
    const totalImages = document.getElementById('totalImages');
    const normalImages = document.getElementById('normalImages');
    const faultyImages = document.getElementById('faultyImages');
    const analyzeAnotherBatchBtn = document.getElementById('analyzeAnotherBatchBtn');
    const downloadBatchReportBtn = document.getElementById('downloadBatchReportBtn');

    // Current analysis data storage
    let currentAnalysis = null;
    let currentBatchAnalysis = null;

    // Handle file selection via button
    uploadBtn.addEventListener('click', () => fileInput.click());
    batchUploadBtn.addEventListener('click', () => batchFileInput.click());

    // Handle file selection
    fileInput.addEventListener('change', handleFiles);
    batchFileInput.addEventListener('change', handleBatchFiles);

    // Drag and drop functionality
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
        batchDropArea.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
        batchDropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
        batchDropArea.addEventListener(eventName, unhighlight, false);
    });

    dropArea.addEventListener('drop', handleDrop, false);
    batchDropArea.addEventListener('drop', handleBatchDrop, false);

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight() {
        this.classList.add('highlight');
    }

    function unhighlight() {
        this.classList.remove('highlight');
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) {
            fileInput.files = files;
            handleFiles({ target: fileInput });
        }
    }

    function handleBatchDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) {
            batchFileInput.files = files;
            handleBatchFiles({ target: batchFileInput });
        }
    }

    // --- Single File Processing ---
    async function handleFiles(e) {
        const file = e.target.files[0];
        if (!file) return;

        if (!file.type.match('image.*')) {
            showAlert('Please select an image file (JPEG, PNG, etc.)', 'error');
            return;
        }

        // Update file info
        fileInfo.innerHTML = `
            <p><strong>Selected file:</strong> ${file.name}</p>
            <p><strong>Size:</strong> ${formatFileSize(file.size)}</p>
        `;
        fileInfo.style.display = 'block';

        loadingOverlay.style.display = 'flex';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();

            // Store analysis data for PDF generation
            currentAnalysis = {
                filename: data.original_image,
                fault_type: data.fault_type,
                confidence: data.confidence,
                recommendation: data.recommendation,
                timestamp: new Date().toISOString()
            };

            // Add to session history
            sessionHistory.add(data);

            // Display results (and push to history)
            displaySingleResult(data, true);
        } catch (error) {
            console.error('Error:', error);
            showAlert(`Error processing image: ${error.message}`, 'error');
        } finally {
            loadingOverlay.style.display = 'none';
        }
    }

    // --- Display Single Result (with history integration) ---
    function displaySingleResult(data, pushToHistory = false) {
        originalImage.src = `/uploads/${data.original_image}`;
        heatmapImage.src = `/uploads/${data.heatmap_image}`;
        faultType.textContent = data.fault_type;
        confidence.textContent = `${data.confidence}%`;
        recommendation.textContent = data.recommendation;
        statusBadge.textContent = data.fault_type === 'Normal' ? 'Normal' : 'Warning';
        statusBadge.className = `status-badge ${data.fault_type === 'Normal' ? 'normal' : 'warning'}`;
        resultsSection.style.display = 'block';
        batchResultsSection.style.display = 'none';
        resultsSection.scrollIntoView({ behavior: 'smooth' });

        if (pushToHistory) {
            history.pushState(
                { type: 'single', data: data },
                '',
                `?analysis=${encodeURIComponent(data.original_image)}`
            );
        }
    }

    // --- Batch File Processing ---
    async function handleBatchFiles(e) {
        const files = e.target.files;
        if (!files || files.length === 0) return;

        const validFiles = Array.from(files).filter(file => file.type.match('image.*'));
        if (validFiles.length === 0) {
            showAlert('Please select image files (JPEG, PNG, etc.)', 'error');
            return;
        }

        // Update file info
        batchFileInfo.innerHTML = `
            <p><strong>Selected files:</strong> ${files.length}</p>
            <p><strong>Total size:</strong> ${formatFileSize(Array.from(files).reduce((total, file) => total + file.size, 0))}</p>
        `;
        batchFileInfo.style.display = 'block';

        loadingOverlay.style.display = 'flex';

        const formData = new FormData();
        Array.from(files).forEach(file => formData.append('files', file));

        try {
            const response = await fetch('/batch_predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();

            // Store batch analysis data for PDF generation
            currentBatchAnalysis = {
                files: Array.from(files).map(file => ({
                    name: file.name,
                    size: file.size,
                    type: file.type
                })),
                results: data.results,
                timestamp: new Date().toISOString()
            };

            // Add each successful result to history
            data.results.filter(r => r.status === 'success').forEach(result => {
                sessionHistory.add(result);
            });

            displayBatchResults(data.results, true);
        } catch (error) {
            console.error('Error:', error);
            showAlert(`Error processing images: ${error.message}`, 'error');
        } finally {
            loadingOverlay.style.display = 'none';
        }
    }

    // --- Display Batch Results (with history integration) ---
    function displayBatchResults(results, pushToHistory = false) {
        resultsGrid.innerHTML = '';
        const total = results.length;
        const normalCount = results.filter(r => r.status === 'success' && r.fault_type === 'Normal').length;
        const faultyCount = results.filter(r => r.status === 'success' && r.fault_type !== 'Normal').length;
        const errorCount = results.filter(r => r.status === 'error').length;
        
        totalImages.textContent = total;
        normalImages.textContent = normalCount;
        faultyImages.textContent = faultyCount;

        results.forEach(result => {
            if (result.status === 'error') {
                resultsGrid.innerHTML += createErrorCard(result);
                return;
            }
            resultsGrid.innerHTML += createResultCard(result);
        });

        batchResultsSection.style.display = 'block';
        resultsSection.style.display = 'none';
        batchResultsSection.scrollIntoView({ behavior: 'smooth' });

        if (pushToHistory) {
            history.pushState(
                { type: 'batch', data: results },
                '',
                '?batch'
            );
        }
    }

    function createErrorCard(result) {
        return `
            <div class="result-card error">
                <div class="card-header">
                    <h4>${result.filename}</h4>
                    <span class="status-badge error">Error</span>
                </div>
                <div class="card-body">
                    <p>${result.error}</p>
                </div>
            </div>
        `;
    }

    function createResultCard(result) {
        return `
            <div class="result-card ${result.fault_type === 'Normal' ? 'normal' : 'warning'}">
                <div class="card-header">
                    <h4>${result.filename}</h4>
                    <span class="status-badge ${result.fault_type === 'Normal' ? 'normal' : 'warning'}">
                        ${result.fault_type === 'Normal' ? 'Normal' : 'Warning'}
                    </span>
                </div>
                <div class="card-body">
                    <div class="image-comparison">
                        <div class="image-container">
                            <img src="/uploads/${result.original_image}" alt="Original">
                        </div>
                        <div class="image-container">
                            <img src="/uploads/${result.heatmap_image}" alt="Analysis">
                        </div>
                    </div>
                    <div class="result-details">
                        <div class="detail-item">
                            <span class="label">Fault Type:</span>
                            <span class="value">${result.fault_type}</span>
                        </div>
                        <div class="detail-item">
                            <span class="label">Confidence:</span>
                            <span class="value">${result.confidence}%</span>
                        </div>
                        <div class="detail-item full-width">
                            <span class="label">Recommd:</span>
                            <span class="value">${result.recommendation}</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    // Download single report
    downloadReportBtn.addEventListener('click', async () => {
        if (!currentAnalysis) {
            showAlert('No analysis results available to download', 'error');
            return;
        }
        loadingOverlay.style.display = 'flex';
        try {
            const response = await fetch('/download_report', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    filename: currentAnalysis.filename,
                    fault_type: currentAnalysis.fault_type,
                    confidence: currentAnalysis.confidence,
                    recommendation: currentAnalysis.recommendation
                })
            });

            if (!response.ok) {
                throw new Error('Failed to generate PDF');
            }

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `solar_report_${currentAnalysis.filename.replace(/\.[^/.]+$/, "")}.pdf`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            a.remove();
        } catch (error) {
            console.error('Error:', error);
            showAlert(`Error generating PDF: ${error.message}`, 'error');
        } finally {
            loadingOverlay.style.display = 'none';
        }
    });

    // Download batch report
    downloadBatchReportBtn.addEventListener('click', async () => {
        if (!currentBatchAnalysis) {
            showAlert('No batch analysis results available to download', 'error');
            return;
        }
        loadingOverlay.style.display = 'flex';
        try {
            const response = await fetch('/download_batch_report', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(currentBatchAnalysis.results)
            });

            if (!response.ok) {
                throw new Error('Failed to generate batch PDF');
            }

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `solar_batch_report_${new Date().toISOString().split('T')[0]}.pdf`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            a.remove();
        } catch (error) {
            console.error('Error:', error);
            showAlert(`Error generating batch PDF: ${error.message}`, 'error');
        } finally {
            loadingOverlay.style.display = 'none';
        }
    });

    // Analyze another buttons
    analyzeAnotherBtn.addEventListener('click', () => {
        resultsSection.style.display = 'none';
        fileInput.value = '';
        fileInfo.style.display = 'none';
    });

    analyzeAnotherBatchBtn.addEventListener('click', () => {
        batchResultsSection.style.display = 'none';
        batchFileInput.value = '';
        batchFileInfo.style.display = 'none';
    });

    // Helper functions
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2) + ' ' + sizes[i]);
    }

    function showAlert(message, type = 'info') {
        // In a real implementation, you might use a more sophisticated alert system
        alert(`${type.toUpperCase()}: ${message}`);
    }

    // Initialize history display
    sessionHistory.updateDisplay();

    // Add clear history button handler
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');
    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', () => sessionHistory.clear());
    }

    // --- Browser History Integration ---
    window.addEventListener('popstate', function(event) {
        if (event.state) {
            if (event.state.type === 'single') {
                displaySingleResult(event.state.data, false);
            } else if (event.state.type === 'batch') {
                displayBatchResults(event.state.data, false);
            }
        } else {
            // Hide results when navigating back to initial state
            resultsSection.style.display = 'none';
            batchResultsSection.style.display = 'none';
        }
    });

    // Parse URL on initial load
    function parseInitialURL() {
        const urlParams = new URLSearchParams(window.location.search);
        if (urlParams.has('analysis')) {
            // Look for analysis in history
            const analysisId = urlParams.get('analysis');
            const foundItem = sessionHistory.items.find(item => 
                item.original_image === decodeURIComponent(analysisId)
            );
            if (foundItem) {
                sessionHistory.load(sessionHistory.items.indexOf(foundItem), false);
                switchTab(singleTabBtn);
            }
        } else if (urlParams.has('batch')) {
            // Handle batch view
            if (sessionHistory.items.length > 0) {
                // In a real app, you might want to store batch results separately
                switchTab(batchTabBtn);
            }
        }
    }

    // Call URL parser on load
    parseInitialURL();
});

// Make sessionHistory available globally
window.sessionHistory = {
    items: [],
    maxItems: 10,

    add: function(analysisData) {
        analysisData.timestamp = new Date().toISOString();
        this.items.unshift(analysisData);
        if (this.items.length > this.maxItems) {
            this.items = this.items.slice(0, this.maxItems);
        }
        this.updateDisplay();
    },

    updateDisplay: function() {
        const historyContainer = document.getElementById('historyContainer');
        if (!historyContainer) return;
        
        if (this.items.length === 0) {
            historyContainer.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-inbox"></i>
                    <h3>No Analysis History Yet</h3>
                    <p>Your analyzed images will appear here for future reference</p>
                </div>
            `;
            return;
        }

        historyContainer.innerHTML = this.items.map((item, index) => `
            <div class="history-item">
                <div class="history-images">
                    <img src="/uploads/${item.original_image}" alt="Original">
                    <img src="/uploads/${item.heatmap_image}" alt="Analysis">
                </div>
                <div class="history-details">
                    <h4>Analysis ${index + 1}</h4>
                    <p><strong>Type:</strong> ${item.fault_type}</p>
                    <p><strong>Confidence:</strong> ${item.confidence}%</p>
                    <div class="history-actions">
                        <button class="btn secondary-btn history-btn" onclick="sessionHistory.load(${index})">
                            <i class="fas fa-eye"></i> View
                        </button>
                        <button class="btn danger-btn history-btn delete-btn" onclick="sessionHistory.remove(${index})">
                            <i class="fas fa-trash"></i> Delete
                        </button>
                    </div>
                </div>
            </div>
        `).join('');
    },

    load: function(index, pushToHistory = true) {
        if (index >= 0 && index < this.items.length) {
            const item = this.items[index];
            // We need to get the displaySingleResult function from the DOMContentLoaded scope
            const displayFunc = window.displaySingleResult || (() => {});
            displayFunc(item, pushToHistory);
            window.scrollTo({ top: 0, behavior: 'smooth' });
            
            // Switch to single analysis tab
            document.getElementById('singleTabBtn').click();
            
            if (pushToHistory) {
                history.pushState(
                    { type: 'single', data: item },
                    '',
                    `?analysis=${encodeURIComponent(item.original_image)}`
                );
            }
        }
    },

    remove: function(index) {
        if (index >= 0 && index < this.items.length) {
            if (confirm('Are you sure you want to delete this analysis from history?')) {
                this.items.splice(index, 1);
                this.updateDisplay();
            }
        }
    },

    clear: function() {
        if (this.items.length === 0) return;
        if (confirm('Are you sure you want to clear all analysis history?')) {
            this.items = [];
            this.updateDisplay();
        }
    }
};

// Make displaySingleResult available globally for history items
window.displaySingleResult = function(data, pushToHistory = false) {
    const originalImage = document.getElementById('originalImage');
    const heatmapImage = document.getElementById('heatmapImage');
    const faultType = document.getElementById('faultType');
    const confidence = document.getElementById('confidence');
    const recommendation = document.getElementById('recommendation');
    const statusBadge = document.getElementById('statusBadge');
    const resultsSection = document.getElementById('resultsSection');
    const batchResultsSection = document.getElementById('batchResultsSection');

    if (!originalImage || !resultsSection) return;

    originalImage.src = `/uploads/${data.original_image}`;
    heatmapImage.src = `/uploads/${data.heatmap_image}`;
    faultType.textContent = data.fault_type;
    confidence.textContent = `${data.confidence}%`;
    recommendation.textContent = data.recommendation;
    statusBadge.textContent = data.fault_type === 'Normal' ? 'Normal' : 'Warning';
    statusBadge.className = `status-badge ${data.fault_type === 'Normal' ? 'normal' : 'warning'}`;
    resultsSection.style.display = 'block';
    batchResultsSection.style.display = 'none';
    resultsSection.scrollIntoView({ behavior: 'smooth' });

    if (pushToHistory) {
        history.pushState(
            { type: 'single', data: data },
            '',
            `?analysis=${encodeURIComponent(data.original_image)}`
        );
    }
};