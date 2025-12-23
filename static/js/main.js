/**
 * MindWatch Core JavaScript
 */

const App = {
    // API utility
    api: {
        baseUrl: '/api',
        
        async request(endpoint, options = {}) {
            const url = `${this.baseUrl}${endpoint}`;
            const headers = {
                'Content-Type': 'application/json',
                ...options.headers
            };
            
            try {
                const response = await fetch(url, {
                    ...options,
                    headers
                });
                
                if (response.status === 401) {
                    // Unauthorized - redirect to login
                    window.location.href = '/login';
                    return null;
                }
                
                const data = await response.json();
                
                if (!response.ok) {
                    throw new Error(data.error || 'Something went wrong');
                }
                
                return data;
            } catch (error) {
                App.ui.toast(error.message, 'error');
                throw error;
            }
        },
        
        get(endpoint) {
            return this.request(endpoint, { method: 'GET' });
        },
        
        post(endpoint, data) {
            return this.request(endpoint, {
                method: 'POST',
                body: JSON.stringify(data)
            });
        },
        
        put(endpoint, data) {
            return this.request(endpoint, {
                method: 'PUT',
                body: JSON.stringify(data)
            });
        },
        
        delete(endpoint) {
            return this.request(endpoint, { method: 'DELETE' });
        }
    },
    
    // UI utilities
    ui: {
        toast(message, type = 'info') {
            const container = document.getElementById('toast-container') || this.createToastContainer();
            
            const toast = document.createElement('div');
            toast.className = `toast toast-${type} fade-in`;
            
            const icon = type === 'success' ? 'check_circle' : 
                        type === 'error' ? 'error' : 'info';
            
            toast.innerHTML = `
                <span class="material-icons">${icon}</span>
                <span>${message}</span>
            `;
            
            container.appendChild(toast);
            
            setTimeout(() => {
                toast.style.opacity = '0';
                setTimeout(() => toast.remove(), 300);
            }, 3000);
        },
        
        createToastContainer() {
            const container = document.createElement('div');
            container.id = 'toast-container';
            container.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 9999;
                display: flex;
                flex-direction: column;
                gap: 10px;
            `;
            document.body.appendChild(container);
            
            // Add toast CSS if not exists
            const style = document.createElement('style');
            style.innerHTML = `
                .toast {
                    background: rgba(30, 41, 59, 0.95);
                    color: white;
                    padding: 12px 20px;
                    border-radius: 8px;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                    border-left: 4px solid #4f46e5;
                    transition: opacity 0.3s;
                    min-width: 300px;
                }
                .toast-success { border-color: #10b981; }
                .toast-error { border-color: #ef4444; }
                .toast .material-icons { font-size: 20px; }
                .toast-success .material-icons { color: #10b981; }
                .toast-error .material-icons { color: #ef4444; }
            `;
            document.head.appendChild(style);
            
            return container;
        },
        
        showLoading(element) {
            element.dataset.originalContent = element.innerHTML;
            element.innerHTML = '<span class="loader"></span> Loading...';
            element.disabled = true;
        },
        
        hideLoading(element) {
            element.innerHTML = element.dataset.originalContent;
            element.disabled = false;
        }
    },
    
    // Auth helpers
    auth: {
        async logout() {
            try {
                await App.api.post('/auth/logout');
                window.location.href = '/';
            } catch (error) {
                console.error('Logout failed', error);
            }
        }
    },
    
    // Initialization
    init() {
        // Navbar scroll effect
        window.addEventListener('scroll', () => {
            const navbar = document.querySelector('.navbar');
            if (navbar) {
                if (window.scrollY > 50) {
                    navbar.classList.add('scrolled');
                } else {
                    navbar.classList.remove('scrolled');
                }
            }
        });
        
        // Active link
        const currentPath = window.location.pathname;
        document.querySelectorAll('.nav-link').forEach(link => {
            if (link.getAttribute('href') === currentPath) {
                link.classList.add('active');
            }
        });
    }
};

document.addEventListener('DOMContentLoaded', () => {
    App.init();
});
