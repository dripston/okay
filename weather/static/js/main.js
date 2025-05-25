document.addEventListener('DOMContentLoaded', function() {
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 100,
                    behavior: 'smooth'
                });
            }
        });
    });

    // Add animation class to feature boxes when they come into view
    const featureBoxes = document.querySelectorAll('.feature-box');
    
    const observerOptions = {
        threshold: 0.2
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = 1;
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);
    
    featureBoxes.forEach(box => {
        box.style.opacity = 0;
        box.style.transform = 'translateY(20px)';
        box.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
        observer.observe(box);
    });
});