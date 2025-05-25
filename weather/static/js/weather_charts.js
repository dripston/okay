// Weather charts handling

// Function to update charts
function updateCharts(forecast) {
    // Prepare data for charts
    const dates = forecast.map(day => {
        const date = new Date(day.date);
        return date.toLocaleDateString('en-US', { weekday: 'short', day: 'numeric' });
    });
    
    const temps = {
        max: forecast.map(day => day.tmax),
        min: forecast.map(day => day.tmin),
        avg: forecast.map(day => day.tavg)
    };
    
    const precipitation = forecast.map(day => day.prcp);
    const windSpeeds = forecast.map(day => day.wspd);
    
    // Set up chart theme to match the purple theme
    Chart.defaults.color = '#ffffff';
    Chart.defaults.borderColor = 'rgba(100, 100, 180, 0.2)';
    Chart.defaults.font.family = "'Roboto', 'Helvetica', 'Arial', sans-serif";
    
    // Custom chart theme
    const chartTheme = {
        backgroundColor: 'rgba(30, 30, 45, 0.7)',
        gridColor: 'rgba(100, 100, 180, 0.2)',
        textColor: '#ffffff',
        fontFamily: "'Roboto', 'Helvetica', 'Arial', sans-serif",
        tooltipBackgroundColor: 'rgba(40, 40, 60, 0.9)',
        tooltipBorderColor: 'rgba(120, 120, 255, 0.4)',
        legendTextColor: '#ffffff'
    };
    
    // Temperature chart
    createTemperatureChart(dates, temps, chartTheme);
    
    // Precipitation chart
    createPrecipitationChart(dates, precipitation, chartTheme);
    
    // Wind speed chart
    createWindChart(dates, windSpeeds, chartTheme);
}

// Function to create temperature chart
function createTemperatureChart(dates, temps, chartTheme) {
    const tempCanvas = document.getElementById('temperature-chart');
    if (!tempCanvas) {
        console.warn('Temperature chart canvas not found');
        return;
    }
    
    const tempCtx = tempCanvas.getContext('2d');
    if (window.tempChart) {
        window.tempChart.destroy();
    }
    
    // Create gradient for temperature chart
    const tempGradient = tempCtx.createLinearGradient(0, 0, 0, 400);
    tempGradient.addColorStop(0, 'rgba(255, 99, 132, 0.2)');
    tempGradient.addColorStop(1, 'rgba(255, 99, 132, 0.05)');
    
    const avgTempGradient = tempCtx.createLinearGradient(0, 0, 0, 400);
    avgTempGradient.addColorStop(0, 'rgba(120, 120, 255, 0.2)');
    avgTempGradient.addColorStop(1, 'rgba(120, 120, 255, 0.05)');
    
    const minTempGradient = tempCtx.createLinearGradient(0, 0, 0, 400);
    minTempGradient.addColorStop(0, 'rgba(75, 192, 192, 0.2)');
    minTempGradient.addColorStop(1, 'rgba(75, 192, 192, 0.05)');
    
    window.tempChart = new Chart(tempCtx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [{
                label: 'Max Temp',
                data: temps.max,
                borderColor: 'rgba(255, 99, 132, 1)',
                backgroundColor: tempGradient,
                tension: 0.3,
                fill: true,
                pointBackgroundColor: temps.max.map(temp => {
                    if (temp > 35) return 'rgba(255, 0, 0, 1)';
                    if (temp > 30) return 'rgba(255, 99, 132, 1)';
                    return 'rgba(255, 159, 64, 1)';
                }),
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(255, 99, 132, 1)',
                pointRadius: 4,
                pointHoverRadius: 6
            }, {
                label: 'Avg Temp',
                data: temps.avg,
                borderColor: 'rgba(120, 120, 255, 1)',
                backgroundColor: avgTempGradient,
                tension: 0.3,
                fill: true,
                pointBackgroundColor: 'rgba(120, 120, 255, 1)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(120, 120, 255, 1)',
                pointRadius: 4,
                pointHoverRadius: 6
            }, {
                label: 'Min Temp',
                data: temps.min,
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: minTempGradient,
                tension: 0.3,
                fill: true,
                pointBackgroundColor: 'rgba(75, 192, 192, 1)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(75, 192, 192, 1)',
                pointRadius: 4,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: chartTheme.textColor,
                        font: {
                            family: chartTheme.fontFamily
                        },
                        boxWidth: 15,
                        usePointStyle: true,
                        pointStyle: 'circle'
                    }
                },
                title: {
                    display: true,
                    text: 'Temperature Forecast',
                    color: chartTheme.textColor,
                    font: {
                        family: chartTheme.fontFamily,
                        size: 16,
                        weight: 'bold'
                    }
                },
                tooltip: {
                    backgroundColor: chartTheme.tooltipBackgroundColor,
                    borderColor: chartTheme.tooltipBorderColor,
                    borderWidth: 1,
                    titleColor: chartTheme.textColor,
                    bodyColor: chartTheme.textColor,
                    titleFont: {
                        family: chartTheme.fontFamily,
                        weight: 'bold'
                    },
                    bodyFont: {
                        family: chartTheme.fontFamily
                    },
                    callbacks: {
                        label: function(context) {
                            const temp = context.parsed.y;
                            let description = '';
                            if (temp > 35) description = ' (Extreme Heat)';
                            else if (temp > 30) description = ' (Hot)';
                            else if (temp > 25) description = ' (Warm)';
                            else if (temp > 20) description = ' (Comfortable)';
                            else if (temp > 15) description = ' (Cool)';
                            else if (temp > 10) description = ' (Cold)';
                            else description = ' (Very Cold)';
                            
                            return context.dataset.label + ': ' + temp.toFixed(1) + '°C' + description;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    title: {
                        display: true,
                        text: 'Temperature (°C)',
                        color: chartTheme.textColor,
                        font: {
                            family: chartTheme.fontFamily
                        }
                    },
                    grid: {
                        color: chartTheme.gridColor
                    },
                    ticks: {
                        color: chartTheme.textColor,
                        font: {
                            family: chartTheme.fontFamily
                        }
                    }
                },
                x: {
                    grid: {
                        color: chartTheme.gridColor
                    },
                    ticks: {
                        color: chartTheme.textColor,
                        font: {
                            family: chartTheme.fontFamily
                        }
                    }
                }
            }
        }
    });
}

// Function to create precipitation chart
function createPrecipitationChart(dates, precipitation, chartTheme) {
    const precipCanvas = document.getElementById('precipitation-chart');
    if (!precipCanvas) {
        console.warn('Precipitation chart canvas not found');
        return;
    }
    
    const precipCtx = precipCanvas.getContext('2d');
    if (window.precipChart) {
        window.precipChart.destroy();
    }
    
    // Create gradient for precipitation chart
    const precipGradient = precipCtx.createLinearGradient(0, 0, 0, 400);
    precipGradient.addColorStop(0, 'rgba(54, 162, 235, 0.5)');
    precipGradient.addColorStop(1, 'rgba(54, 162, 235, 0.1)');
    
    window.precipChart = new Chart(precipCtx, {
        type: 'bar',
        data: {
            labels: dates,
            datasets: [{
                label: 'Precipitation',
                data: precipitation,
                backgroundColor: precipitation.map(prcp => {
                    if (prcp > 10) return 'rgba(54, 162, 235, 0.9)';
                    if (prcp > 5) return 'rgba(54, 162, 235, 0.7)';
                    if (prcp > 0) return 'rgba(54, 162, 235, 0.5)';
                    return 'rgba(54, 162, 235, 0.2)';
                }),
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1,
                borderRadius: 5,
                hoverBackgroundColor: 'rgba(54, 162, 235, 1)'
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: chartTheme.textColor,
                        font: {
                            family: chartTheme.fontFamily
                        },
                        boxWidth: 15,
                        usePointStyle: true,
                        pointStyle: 'rect'
                    }
                },
                title: {
                    display: true,
                    text: 'Precipitation Forecast',
                    color: chartTheme.textColor,
                    font: {
                        family: chartTheme.fontFamily,
                        size: 16,
                        weight: 'bold'
                    }
                },
                tooltip: {
                    backgroundColor: chartTheme.tooltipBackgroundColor,
                    borderColor: chartTheme.tooltipBorderColor,
                    borderWidth: 1,
                    titleColor: chartTheme.textColor,
                    bodyColor: chartTheme.textColor,
                    titleFont: {
                        family: chartTheme.fontFamily,
                        weight: 'bold'
                    },
                    bodyFont: {
                        family: chartTheme.fontFamily
                    },
                    callbacks: {
                        label: function(context) {
                            const prcp = context.parsed.y;
                            let description = '';
                            if (prcp > 10) description = ' (Heavy Rain)';
                            else if (prcp > 5) description = ' (Moderate Rain)';
                            else if (prcp > 0) description = ' (Light Rain)';
                            else description = ' (No Rain)';
                            
                            return context.dataset.label + ': ' + prcp.toFixed(1) + ' mm' + description;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Precipitation (mm)',
                        color: chartTheme.textColor,
                        font: {
                            family: chartTheme.fontFamily
                        }
                    },
                    grid: {
                        color: chartTheme.gridColor
                    },
                    ticks: {
                        color: chartTheme.textColor,
                        font: {
                            family: chartTheme.fontFamily
                        }
                    }
                },
                x: {
                    grid: {
                        color: chartTheme.gridColor
                    },
                    ticks: {
                        color: chartTheme.textColor,
                        font: {
                            family: chartTheme.fontFamily
                        }
                    }
                }
            }
        }
    });
}

// Function to create wind speed chart
function createWindChart(dates, windSpeeds, chartTheme) {
    const windCanvas = document.getElementById('wind-chart');
    if (!windCanvas) {
        console.warn('Wind chart canvas not found');
        return;
    }
    
    const windCtx = windCanvas.getContext('2d');
    if (window.windChart) {
        window.windChart.destroy();
    }
    
    // Create gradient for wind chart
    const windGradient = windCtx.createLinearGradient(0, 0, 0, 400);
    windGradient.addColorStop(0, 'rgba(255, 159, 64, 0.3)');
    windGradient.addColorStop(1, 'rgba(255, 159, 64, 0.05)');
    
    window.windChart = new Chart(windCtx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [{
                label: 'Wind Speed',
                data: windSpeeds,
                borderColor: 'rgba(255, 159, 64, 1)',
                backgroundColor: windGradient,
                tension: 0.3,
                fill: true,
                pointBackgroundColor: windSpeeds.map(speed => {
                    if (speed > 20) return 'rgba(255, 99, 132, 1)';
                    if (speed > 10) return 'rgba(255, 159, 64, 1)';
                    return 'rgba(75, 192, 192, 1)';
                }),
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(255, 159, 64, 1)',
                pointRadius: 4,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: chartTheme.textColor,
                        font: {
                            family: chartTheme.fontFamily
                        },
                        boxWidth: 15,
                        usePointStyle: true,
                        pointStyle: 'circle'
                    }
                },
                title: {
                    display: true,
                    text: 'Wind Speed Forecast',
                    color: chartTheme.textColor,
                    font: {
                        family: chartTheme.fontFamily,
                        size: 16,
                        weight: 'bold'
                    }
                },
                tooltip: {
                    backgroundColor: chartTheme.tooltipBackgroundColor,
                    borderColor: chartTheme.tooltipBorderColor,
                    borderWidth: 1,
                    titleColor: chartTheme.textColor,
                    bodyColor: chartTheme.textColor,
                    titleFont: {
                        family: chartTheme.fontFamily,
                        weight: 'bold'
                    },
                    bodyFont: {
                        family: chartTheme.fontFamily
                    },
                    callbacks: {
                        label: function(context) {
                            const speed = context.parsed.y;
                            let description = '';
                            if (speed > 20) description = ' (Strong/Gusty)';
                            else if (speed > 10) description = ' (Moderate)';
                            else description = ' (Light)';
                            
                            return context.dataset.label + ': ' + speed.toFixed(1) + ' m/s' + description;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Wind Speed (m/s)',
                        color: chartTheme.textColor,
                        font: {
                            family: chartTheme.fontFamily
                        }
                    },
                    grid: {
                        color: chartTheme.gridColor
                    },
                    ticks: {
                        color: chartTheme.textColor,
                        font: {
                            family: chartTheme.fontFamily
                        }
                    }
                },
                x: {
                    grid: {
                        color: chartTheme.gridColor
                    },
                    ticks: {
                        color: chartTheme.textColor,
                        font: {
                            family: chartTheme.fontFamily
                        }
                    }
                }
            }
        }
    });
}

// Function to clear charts
function clearCharts() {
    if (window.tempChart) {
        window.tempChart.destroy();
        window.tempChart = null;
    }
    
    if (window.precipChart) {
        window.precipChart.destroy();
        window.precipChart = null;
    }
    
    if (window.windChart) {
        window.windChart.destroy();
        window.windChart = null;
    }
}

// Export functions
window.weatherCharts = {
    updateCharts,
    createTemperatureChart,
    createPrecipitationChart,
    createWindChart,
    clearCharts
};