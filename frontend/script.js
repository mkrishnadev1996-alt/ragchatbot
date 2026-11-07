// API base URL - use relative path to work from any host
const API_URL = '/api';

// Global state
let currentSessionId = null;
let authToken = null;
let authMode = 'login'; // 'login' | 'register'

const TOKEN_KEY = 'authToken';

// localStorage can throw (private mode, blocked site data), so never rely on it
function loadToken() {
    try { return localStorage.getItem(TOKEN_KEY); } catch (_) { return null; }
}

function saveToken(token) {
    authToken = token;
    try {
        if (token) localStorage.setItem(TOKEN_KEY, token);
        else localStorage.removeItem(TOKEN_KEY);
    } catch (_) { /* token just lives for this page view */ }
}

// DOM elements
let chatMessages, chatInput, sendButton, totalCourses, courseTitles;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Get DOM elements after page loads
    chatMessages = document.getElementById('chatMessages');
    chatInput = document.getElementById('chatInput');
    sendButton = document.getElementById('sendButton');
    totalCourses = document.getElementById('totalCourses');
    courseTitles = document.getElementById('courseTitles');
    
    setupEventListeners();
    setupAuth();
    createNewSession();
    loadCourseStats();
});

// Auth
function setupAuth() {
    document.getElementById('authForm').addEventListener('submit', submitAuth);
    document.getElementById('authSwitch').addEventListener('click', () => {
        setAuthMode(authMode === 'login' ? 'register' : 'login');
    });
    document.getElementById('logoutButton').addEventListener('click', logout);

    authToken = loadToken();
    if (authToken) {
        // Validate the stored token; falls back to the login modal if it's stale
        authFetch(`${API_URL}/auth/me`)
            .then(res => res.ok ? res.json() : null)
            .then(user => { if (user) showUser(user); })
            .catch(() => showAuthModal());
    } else {
        showAuthModal();
    }
}

function setAuthMode(mode) {
    authMode = mode;
    const login = mode === 'login';
    document.getElementById('authTitle').textContent = login ? 'Log in' : 'Create account';
    document.getElementById('authSubmit').textContent = login ? 'Log in' : 'Register';
    document.getElementById('authSwitchText').textContent = login ? 'No account?' : 'Have an account?';
    document.getElementById('authSwitch').textContent = login ? 'Register' : 'Log in';
    document.getElementById('authPassword').autocomplete = login ? 'current-password' : 'new-password';
    setAuthError('');
}

function setAuthError(message) {
    const el = document.getElementById('authError');
    el.textContent = message;
    el.hidden = !message;
}

function showAuthModal() {
    document.getElementById('userMenu').hidden = true;
    document.getElementById('authModal').hidden = false;
    document.getElementById('authEmail').focus();
}

function showUser(user) {
    document.getElementById('authModal').hidden = true;
    document.getElementById('userEmail').textContent = user.email;
    document.getElementById('userMenu').hidden = false;
    chatInput.focus();
    loadHistory();
}

// Replay the user's saved conversation below the welcome message
async function loadHistory() {
    try {
        const response = await authFetch(`${API_URL}/messages`);
        if (!response.ok) return;
        const messages = await response.json();
        for (const m of messages) {
            addMessage(m.content, m.role, m.sources, false, m.message_id, m.rating);
        }
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

async function submitAuth(event) {
    event.preventDefault();
    const email = document.getElementById('authEmail').value.trim();
    const password = document.getElementById('authPassword').value;
    const submit = document.getElementById('authSubmit');

    if (!email || password.length < 8) {
        setAuthError('Enter an email and a password of at least 8 characters.');
        return;
    }

    submit.disabled = true;
    setAuthError('');
    try {
        const response = await fetch(`${API_URL}/auth/${authMode}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password })
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            // 422 detail is a list of validation errors; show a generic message for it
            const detail = typeof data.detail === 'string' ? data.detail : 'Please check your email and password.';
            throw new Error(detail);
        }
        saveToken(data.token);
        document.getElementById('authPassword').value = '';
        createNewSession(); // a new account/login starts a fresh conversation
        showUser(data.user);
    } catch (error) {
        setAuthError(error.message || 'Something went wrong. Please try again.');
    } finally {
        submit.disabled = false;
    }
}

function logout() {
    saveToken(null);
    createNewSession();
    showAuthModal();
}

// fetch wrapper that attaches the bearer token and handles expired sessions
async function authFetch(url, options = {}) {
    const headers = { ...(options.headers || {}) };
    if (authToken) headers['Authorization'] = `Bearer ${authToken}`;
    const response = await fetch(url, { ...options, headers });
    if (response.status === 401) {
        saveToken(null);
        showAuthModal();
    }
    return response;
}

// Event Listeners
function setupEventListeners() {
    // Chat functionality
    sendButton.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
    
    
    // Suggested questions
    document.querySelectorAll('.suggested-item').forEach(button => {
        button.addEventListener('click', (e) => {
            const question = e.target.getAttribute('data-question');
            chatInput.value = question;
            sendMessage();
        });
    });
}


// Chat Functions
async function sendMessage() {
    const query = chatInput.value.trim();
    if (!query) return;

    // Disable input
    chatInput.value = '';
    chatInput.disabled = true;
    sendButton.disabled = true;

    // Add user message
    addMessage(query, 'user');

    // Add loading message - create a unique container for it
    const loadingMessage = createLoadingMessage();
    chatMessages.appendChild(loadingMessage);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        const response = await authFetch(`${API_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                session_id: currentSessionId
            })
        });

        if (!response.ok) {
            let message = 'Query failed';
            // Only surface the server's message for rate limits; other errors stay generic
            if (response.status === 401) {
                message = 'Your session expired. Please log in again.';
            } else if (response.status === 404) {
                // Session unknown to the server (e.g. it restarted): start fresh
                currentSessionId = null;
                message = 'Your conversation was reset. Please ask again.';
            } else if (response.status === 429) {
                message = 'The AI service is busy. Please try again in a moment.';
                try {
                    const err = await response.json();
                    if (err.detail) message = err.detail;
                } catch (_) { /* keep default rate-limit message */ }
            }
            throw new Error(message);
        }

        const data = await response.json();
        
        // Update session ID if new
        if (!currentSessionId) {
            currentSessionId = data.session_id;
        }

        // Replace loading message with response
        loadingMessage.remove();
        addMessage(data.answer, 'assistant', data.sources, false, data.message_id);

    } catch (error) {
        // Replace loading message with error
        loadingMessage.remove();
        addMessage(`Error: ${error.message}`, 'assistant');
    } finally {
        chatInput.disabled = false;
        sendButton.disabled = false;
        chatInput.focus();
    }
}

function createLoadingMessage() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.innerHTML = `
        <div class="message-content">
            <div class="loading">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    return messageDiv;
}

function addMessage(content, type, sources = null, isWelcome = false, feedbackId = null, rating = null) {
    const messageId = Date.now();
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}${isWelcome ? ' welcome-message' : ''}`;
    messageDiv.id = `message-${messageId}`;
    
    // Convert markdown to HTML for assistant messages
    const displayContent = type === 'assistant' ? marked.parse(content) : escapeHtml(content);
    
    let html = `<div class="message-content">${displayContent}</div>`;
    
    if (sources && sources.length > 0) {
        html += `
            <details class="sources-collapsible">
                <summary class="sources-header">Sources</summary>
                <div class="sources-content">${renderSources(sources)}</div>
            </details>
        `;
    }
    
    messageDiv.innerHTML = html;
    if (feedbackId) messageDiv.appendChild(createFeedbackControls(feedbackId, currentSessionId, rating));
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    return messageId;
}

// Thumbs up/down under an answer. Clicking the active thumb again clears the rating.
function createFeedbackControls(feedbackId, sessionId, initialRating = null) {
    const bar = document.createElement('div');
    bar.className = 'feedback';
    bar.setAttribute('role', 'group');
    bar.setAttribute('aria-label', 'Rate this answer');

    const buttons = {};
    let current = initialRating;

    const render = () => {
        for (const [rating, button] of Object.entries(buttons)) {
            const active = current === rating;
            button.classList.toggle('active', active);
            button.setAttribute('aria-pressed', String(active));
        }
    };

    const makeButton = (rating, label, icon) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'feedback-button';
        button.title = label;
        button.setAttribute('aria-label', label);
        button.textContent = icon;
        button.addEventListener('click', async () => {
            const previous = current;
            current = current === rating ? null : rating;
            render(); // optimistic; rolled back below if the request fails
            try {
                const response = await authFetch(`${API_URL}/feedback`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message_id: feedbackId,
                        rating: current,
                        session_id: sessionId
                    })
                });
                if (!response.ok) throw new Error('Feedback failed');
            } catch (_) {
                current = previous;
                render();
            }
        });
        buttons[rating] = button;
        return button;
    };

    bar.append(
        makeButton('up', 'Good answer', '👍'),
        makeButton('down', 'Bad answer', '👎')
    );
    render();
    return bar;
}

// Group sources ("Course - Lesson N") by course, dedupe, and render as chips
function renderSources(sources) {
    const groups = new Map();
    for (const source of sources) {
        const idx = source.lastIndexOf(' - Lesson ');
        const course = idx === -1 ? source : source.slice(0, idx);
        const lesson = idx === -1 ? null : source.slice(idx + 3);
        if (!groups.has(course)) groups.set(course, new Set());
        if (lesson) groups.get(course).add(lesson);
    }
    return Array.from(groups, ([course, lessons]) => {
        const chips = Array.from(lessons)
            .sort((a, b) => parseInt(a.replace(/\D/g, '')) - parseInt(b.replace(/\D/g, '')))
            .map(l => `<span class="source-chip">${escapeHtml(l)}</span>`)
            .join('');
        return `<div class="source-group">
            <div class="source-course">${escapeHtml(course)}</div>
            ${chips ? `<div class="source-lessons">${chips}</div>` : ''}
        </div>`;
    }).join('');
}

// Helper function to escape HTML for user messages
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Removed removeMessage function - no longer needed since we handle loading differently

async function createNewSession() {
    currentSessionId = null;
    chatMessages.innerHTML = '';
    addMessage('Welcome to the Course Materials Assistant! I can help you with questions about courses, lessons and specific content. What would you like to know?', 'assistant', null, true);
}

// Load course statistics
async function loadCourseStats() {
    try {
        console.log('Loading course stats...');
        const response = await fetch(`${API_URL}/courses`);
        if (!response.ok) throw new Error('Failed to load course stats');
        
        const data = await response.json();
        console.log('Course data received:', data);
        
        // Update stats in UI
        if (totalCourses) {
            totalCourses.textContent = data.total_courses;
        }
        
        // Update course titles
        if (courseTitles) {
            if (data.course_titles && data.course_titles.length > 0) {
                courseTitles.innerHTML = data.course_titles
                    .map(title => `<div class="course-title-item">${escapeHtml(title)}</div>`)
                    .join('');
            } else {
                courseTitles.innerHTML = '<span class="no-courses">No courses available</span>';
            }
        }
        
    } catch (error) {
        console.error('Error loading course stats:', error);
        // Set default values on error
        if (totalCourses) {
            totalCourses.textContent = '0';
        }
        if (courseTitles) {
            courseTitles.innerHTML = '<span class="error">Failed to load courses</span>';
        }
    }
}