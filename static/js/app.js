// Facial Emotion Recognition App - Frontend JavaScript

// State management
const state = {
    currentTab: 'upload',
    webcamStream: null,
    webcamActive: false
};

// DOM Elements
const elements = {
    // Tabs
    tabButtons: document.querySelectorAll('.tab-button'),
    tabContents: document.querySelectorAll('.tab-content'),

    // Upload
    dropZone: document.getElementById('dropZone'),
    fileInput: document.getElementById('fileInput'),
    resultSection: document.getElementById('resultSection'),
    resultImage: document.getElementById('resultImage'),
    emotionResults: document.getElementById('emotionResults'),

    // Webcam
    webcam: document.getElementById('webcam'),
    canvas: document.getElementById('canvas'),
    startWebcamBtn: document.getElementById('startWebcam'),
    captureBtn: document.getElementById('captureBtn'),
    stopWebcamBtn: document.getElementById('stopWebcam'),
    webcamResultSection: document.getElementById('webcamResultSection'),
    webcamResultImage: document.getElementById('webcamResultImage'),
    webcamEmotionResults: document.getElementById('webcamEmotionResults'),

    // Statistics
    totalUploads: document.getElementById('totalUploads'),
    emotionStats: document.getElementById('emotionStats'),
    refreshStatsBtn: document.getElementById('refreshStats'),

    // UI
    loadingOverlay: document.getElementById('loadingOverlay'),
    toast: document.getElementById('toast')
};

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    initializeTabs();
    initializeUpload();
    initializeWebcam();
    initializeStats();
});

// Tab Management
function initializeTabs() {
    elements.tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabName = button.dataset.tab;
            switchTab(tabName);
        });
    });
}

function switchTab(tabName) {
    // Update tab buttons
    elements.tabButtons.forEach(btn => {
        if (btn.dataset.tab === tabName) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });

    // Update tab contents
    elements.tabContents.forEach(content => {
        if (content.id === `${tabName}-tab`) {
            content.classList.add('active');
        } else {
            content.classList.remove('active');
        }
    });

    state.currentTab = tabName;

    // Load stats when switching to stats tab
    if (tabName === 'stats') {
        loadStatistics();
    }
}

// Upload Functionality
function initializeUpload() {
    // File input change
    elements.fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    elements.dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.dropZone.classList.add('drag-over');
    });

    elements.dropZone.addEventListener('dragleave', () => {
        elements.dropZone.classList.remove('drag-over');
    });

    elements.dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.dropZone.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // Click to upload
    elements.dropZone.addEventListener('click', (e) => {
        if (e.target.tagName !== 'INPUT' && e.target.tagName !== 'SPAN') {
            elements.fileInput.click();
        }
    });
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

async function handleFile(file) {
    // Validate file
    if (!file.type.startsWith('image/')) {
        showToast('Please select an image file', 'error');
        return;
    }

    // Check file size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
        showToast('File too large. Maximum size is 16MB', 'error');
        return;
    }

    // Show loading
    showLoading();

    // Create form data
    const formData = new FormData();
    formData.append('file', file);

    try {
        // Upload and process
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            displayUploadResults(data.data);
            showToast('Image processed successfully!', 'success');
        } else {
            showToast(data.error || 'Processing failed', 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showToast('Error processing image', 'error');
    } finally {
        hideLoading();
    }
}

function displayUploadResults(data) {
    // Show result section
    elements.resultSection.classList.remove('hidden');

    // Display annotated image
    elements.resultImage.src = data.annotated_image;

    // Display emotion results
    elements.emotionResults.innerHTML = '';

    if (data.num_faces === 0) {
        elements.emotionResults.innerHTML = '<p>No faces detected in the image.</p>';
    } else {
        data.results.forEach((result, index) => {
            const emotionItem = createEmotionItem(result, index + 1);
            elements.emotionResults.appendChild(emotionItem);
        });
    }

    // Scroll to results
    elements.resultSection.scrollIntoView({ behavior: 'smooth' });
}

// Webcam Functionality
function initializeWebcam() {
    elements.startWebcamBtn.addEventListener('click', startWebcam);
    elements.stopWebcamBtn.addEventListener('click', stopWebcam);
    elements.captureBtn.addEventListener('click', captureAndAnalyze);
}

async function startWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 }
        });

        elements.webcam.srcObject = stream;
        state.webcamStream = stream;
        state.webcamActive = true;

        // Update UI
        elements.startWebcamBtn.classList.add('hidden');
        elements.captureBtn.classList.remove('hidden');
        elements.stopWebcamBtn.classList.remove('hidden');

        showToast('Camera started', 'success');
    } catch (error) {
        console.error('Error accessing webcam:', error);
        showToast('Could not access camera', 'error');
    }
}

function stopWebcam() {
    if (state.webcamStream) {
        state.webcamStream.getTracks().forEach(track => track.stop());
        elements.webcam.srcObject = null;
        state.webcamStream = null;
        state.webcamActive = false;

        // Update UI
        elements.startWebcamBtn.classList.remove('hidden');
        elements.captureBtn.classList.add('hidden');
        elements.stopWebcamBtn.classList.add('hidden');

        showToast('Camera stopped', 'success');
    }
}

async function captureAndAnalyze() {
    if (!state.webcamActive) {
        showToast('Camera is not active', 'error');
        return;
    }

    // Set canvas size to match video
    elements.canvas.width = elements.webcam.videoWidth;
    elements.canvas.height = elements.webcam.videoHeight;

    // Draw video frame to canvas
    const ctx = elements.canvas.getContext('2d');
    ctx.drawImage(elements.webcam, 0, 0);

    // Convert to base64
    const base64Image = elements.canvas.toDataURL('image/jpeg').split(',')[1];

    // Show loading
    showLoading();

    try {
        // Send to backend
        const response = await fetch('/webcam', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: base64Image })
        });

        const data = await response.json();

        if (data.success) {
            displayWebcamResults(data.data);
            showToast('Image analyzed successfully!', 'success');
        } else {
            showToast(data.error || 'Analysis failed', 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showToast('Error analyzing image', 'error');
    } finally {
        hideLoading();
    }
}

function displayWebcamResults(data) {
    // Show result section
    elements.webcamResultSection.classList.remove('hidden');

    // Display annotated image
    elements.webcamResultImage.src = data.annotated_image;

    // Display emotion results
    elements.webcamEmotionResults.innerHTML = '';

    if (data.num_faces === 0) {
        elements.webcamEmotionResults.innerHTML = '<p>No faces detected in the image.</p>';
    } else {
        data.results.forEach((result, index) => {
            const emotionItem = createEmotionItem(result, index + 1);
            elements.webcamEmotionResults.appendChild(emotionItem);
        });
    }

    // Scroll to results
    elements.webcamResultSection.scrollIntoView({ behavior: 'smooth' });
}

// Statistics Functionality
function initializeStats() {
    elements.refreshStatsBtn.addEventListener('click', loadStatistics);
}

async function loadStatistics() {
    showLoading();

    try {
        const response = await fetch('/statistics');
        const data = await response.json();

        if (data.success) {
            displayStatistics(data.data);
        } else {
            showToast('Failed to load statistics', 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showToast('Error loading statistics', 'error');
    } finally {
        hideLoading();
    }
}

function displayStatistics(stats) {
    // Display total uploads
    elements.totalUploads.textContent = stats.total_uploads || 0;

    // Display emotion statistics
    elements.emotionStats.innerHTML = '';

    const emotionData = stats.by_emotion || {};
    const emotions = Object.keys(emotionData);

    if (emotions.length === 0) {
        elements.emotionStats.innerHTML = '<p>No training data available yet.</p>';
    } else {
        emotions.forEach(emotion => {
            const statItem = document.createElement('div');
            statItem.className = 'emotion-stat-item';
            statItem.innerHTML = `
                <div class="emotion-label">${emotion}</div>
                <div class="emotion-count">${emotionData[emotion]}</div>
            `;
            elements.emotionStats.appendChild(statItem);
        });
    }
}

// Helper Functions
function createEmotionItem(result, faceNumber) {
    const item = document.createElement('div');
    item.className = 'emotion-item';

    const confidence = (result.confidence * 100).toFixed(1);

    item.innerHTML = `
        <div class="emotion-name">Face ${faceNumber}: ${result.emotion}</div>
        <div class="emotion-confidence">Confidence: ${confidence}%</div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: ${confidence}%"></div>
        </div>
    `;

    return item;
}

function showLoading() {
    elements.loadingOverlay.classList.remove('hidden');
}

function hideLoading() {
    elements.loadingOverlay.classList.add('hidden');
}

function showToast(message, type = 'info') {
    elements.toast.textContent = message;
    elements.toast.className = 'toast';
    if (type) {
        elements.toast.classList.add(type);
    }
    elements.toast.classList.remove('hidden');

    setTimeout(() => {
        elements.toast.classList.add('hidden');
    }, 3000);
}

// Emotion color mapping
const emotionColors = {
    happy: '#10b981',
    sad: '#3b82f6',
    angry: '#ef4444',
    surprise: '#f59e0b',
    fear: '#8b5cf6',
    disgust: '#ec4899',
    neutral: '#6b7280'
};
