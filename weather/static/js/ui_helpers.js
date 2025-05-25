// UI Helpers for forecast display

// Global variables
let updateInterval = null;
let statusCheckInterval = null;

// Function to show progress bar
function showProgressBar() {
    // Hide regular loading indicator if it exists
    const loadingElement = document.querySelector('.loading');
    if (loadingElement) {
        loadingElement.style.display = 'none';
    }
    
    // Create progress bar container if it doesn't exist
    let progressContainer = document.getElementById('forecast-progress-container');
    if (!progressContainer) {
        progressContainer = document.createElement('div');
        progressContainer.id = 'forecast-progress-container';
        progressContainer.className = 'progress-container';
        progressContainer.innerHTML = `
            <h3><i class="fas fa-cogs"></i> Generating new weather forecast...</h3>
            <div class="progress">
                <div id="forecast-progress-bar" class="progress-bar" role="progressbar" 
                     style="width: 0%;" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100">0%</div>
            </div>
            <p id="forecast-status-message" class="mt-2">Initializing forecast model...</p>
        `;
        
        // Find a suitable place to insert the progress container
        const header = document.querySelector('h1, h2');
        if (header && header.parentNode) {
            header.parentNode.insertBefore(progressContainer, header.nextSibling);
        } else {
            // If no header, insert at the beginning of the main content
            const mainContent = document.querySelector('main, .main-content, #content');
            if (mainContent) {
                mainContent.prepend(progressContainer);
            } else {
                // Last resort: add to body
                document.body.prepend(progressContainer);
            }
        }
    } else {
        // Reset progress bar
        const progressBar = document.getElementById('forecast-progress-bar');
        if (progressBar) {
            progressBar.style.width = '0%';
            progressBar.textContent = '0%';
            progressBar.setAttribute('aria-valuenow', '0');
        }
        
        // Reset status message
        const statusMessage = document.getElementById('forecast-status-message');
        if (statusMessage) {
            statusMessage.textContent = 'Initializing forecast model...';
        }
        
        // Show the container
        progressContainer.style.display = 'block';
    }
}

// Function to hide progress bar
function hideProgressBar() {
    const progressContainer = document.getElementById('forecast-progress-container');
    if (progressContainer) {
        progressContainer.style.display = 'none';
    }
    
    // Clear status check interval
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
        statusCheckInterval = null;
    }
}

// Function to ensure all UI elements exist
function ensureUIElementsExist() {
    // Check for forecast cards container
    let forecastCards = document.getElementById('forecast-cards');
    if (!forecastCards) {
        console.warn('Forecast cards container not found, creating one');
        const forecastContent = document.querySelector('.forecast-content');
        if (forecastContent) {
            forecastCards = document.createElement('div');
            forecastCards.id = 'forecast-cards';
            forecastCards.className = 'forecast-cards';
            
            // Find where to insert it
            const h2 = forecastContent.querySelector('h2');
            if (h2) {
                forecastContent.insertBefore(forecastCards, h2.nextSibling);
            } else {
                forecastContent.appendChild(forecastCards);
            }
        }
    }
    
    // Check for chart containers
    const chartContainers = [
        { id: 'temperature-chart', title: 'Temperature Forecast', icon: 'fas fa-temperature-high' },
        { id: 'precipitation-chart', title: 'Precipitation Forecast', icon: 'fas fa-cloud-rain' },
        { id: 'wind-chart', title: 'Wind Speed Forecast', icon: 'fas fa-wind' }
    ];
    
    let forecastCharts = document.querySelector('.forecast-charts');
    if (!forecastCharts) {
        console.warn('Forecast charts container not found, creating one');
        const forecastContent = document.querySelector('.forecast-content');
        if (forecastContent) {
            forecastCharts = document.createElement('div');
            forecastCharts.className = 'forecast-charts';
            forecastContent.appendChild(forecastCharts);
        }
    }
    
    if (forecastCharts) {
        chartContainers.forEach(chart => {
            if (!document.getElementById(chart.id)) {
                console.warn(`Chart container for ${chart.id} not found, creating one`);
                const chartContainer = document.createElement('div');
                chartContainer.className = 'chart-container';
                chartContainer.innerHTML = `
                    <h3><i class="${chart.icon}"></i> ${chart.title}</h3>
                    <canvas id="${chart.id}"></canvas>
                `;
                forecastCharts.appendChild(chartContainer);
            }
        });
    }
    
    // Check for weather alerts container
    let alertsContainer = document.getElementById('weather-alerts-container');
    if (!alertsContainer) {
        console.warn('Weather alerts container not found, creating one');
        const alertsSection = document.querySelector('.weather-alerts');
        if (alertsSection) {
            alertsContainer = document.createElement('div');
            alertsContainer.id = 'weather-alerts-container';
            alertsSection.appendChild(alertsContainer);
        } else {
            // Create the entire section if it doesn't exist
            const mainContent = document.querySelector('.container');
            if (mainContent) {
                const alertsSection = document.createElement('section');
                alertsSection.className = 'weather-alerts';
                alertsSection.innerHTML = `
                    <h2><i class="fas fa-exclamation-triangle"></i> Weather Alerts</h2>
                `;
                alertsContainer = document.createElement('div');
                alertsContainer.id = 'weather-alerts-container';
                alertsSection.appendChild(alertsContainer);
                
                // Insert before footer if it exists
                const footer = document.querySelector('footer');
                if (footer) {
                    mainContent.insertBefore(alertsSection, footer);
                } else {
                    mainContent.appendChild(alertsSection);
                }
            }
        }
    }
}

// Function to add refresh button
function addRefreshButton(fetchCallback) {
    let refreshButton = document.getElementById('refresh-forecast-btn');
    if (!refreshButton) {
        const forecastHeader = document.querySelector('.forecast-content h2');
        if (forecastHeader) {
            refreshButton = document.createElement('button');
            refreshButton.id = 'refresh-forecast-btn';
            refreshButton.className = 'btn btn-sm btn-primary refresh-btn';
            refreshButton.innerHTML = '<i class="fas fa-sync-alt"></i> Refresh';
            forecastHeader.appendChild(refreshButton);
        }
    }
    
    if (refreshButton) {
        // Remove existing event listeners
        const newButton = refreshButton.cloneNode(true);
        if (refreshButton.parentNode) {
            refreshButton.parentNode.replaceChild(newButton, refreshButton);
        }
        
        // Add new event listener
        newButton.addEventListener('click', () => {
            newButton.disabled = true;
            newButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Updating...';
            
            // Show loading indicator
            const forecastCards = document.getElementById('forecast-cards');
            if (forecastCards) {
                forecastCards.innerHTML = `
                    <div class="loading">
                        <div class="loading-spinner"></div>
                        <p>Refreshing forecast data...</p>
                    </div>
                `;
            }
            
            // Fetch new data
            fetchCallback().then(() => {
                newButton.disabled = false;
                newButton.innerHTML = '<i class="fas fa-sync-alt"></i> Refresh';
            }).catch(() => {
                newButton.disabled = false;
                newButton.innerHTML = '<i class="fas fa-sync-alt"></i> Refresh';
            });
        });
    }
}

// Function to start countdown for next update
function startUpdateCountdown(nextUpdateTime, fetchCallback) {
    const countdownElement = document.getElementById('next-update-countdown');
    if (!countdownElement) return;
    
    // Clear any existing interval
    if (updateInterval) {
        clearInterval(updateInterval);
    }
    
    // Update the countdown every second
    updateInterval = setInterval(() => {
        const now = new Date();
        const diff = nextUpdateTime - now;
        
        if (diff <= 0) {
            // Time's up, clear interval and refresh data
            clearInterval(updateInterval);
            countdownElement.textContent = 'Updating...';
            fetchCallback();
            return;
        }
        
        // Calculate minutes and seconds
        const minutes = Math.floor(diff / 60000);
        const seconds = Math.floor((diff % 60000) / 1000);
        
        // Display the countdown
        countdownElement.textContent = `${minutes}m ${seconds}s`;
    }, 1000);
}

// Function to update last update time
function updateLastUpdateTime(timestamp) {
    const lastUpdateElement = document.getElementById('last-update-time');
    if (lastUpdateElement) {
        const date = new Date(timestamp);
        lastUpdateElement.textContent = date.toLocaleString('en-US', {
            weekday: 'short',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    }
}

// Export functions
window.uiHelpers = {
    showProgressBar,
    hideProgressBar,
    ensureUIElementsExist,
    addRefreshButton,
    startUpdateCountdown,
    updateLastUpdateTime
};