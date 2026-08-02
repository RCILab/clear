window.HELP_IMPROVE_VIDEOJS = false;

// More Works Dropdown Functionality
function toggleMoreWorks() {
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (dropdown.classList.contains('show')) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    } else {
        dropdown.classList.add('show');
        button.classList.add('active');
    }
}

// Close dropdown when clicking outside
document.addEventListener('click', function(event) {
    const container = document.querySelector('.more-works-container');
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (container && !container.contains(event.target)) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Close dropdown on escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const dropdown = document.getElementById('moreWorksDropdown');
        const button = document.querySelector('.more-works-btn');
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Copy BibTeX to clipboard
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');
    
    if (bibtexElement) {
        navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
            // Success feedback
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        }).catch(function(err) {
            console.error('Failed to copy: ', err);
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = bibtexElement.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        });
    }
}

// Scroll to top functionality
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Show/hide scroll to top button
window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');
    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }
});

// Play every video while it is on screen, and pause it once it scrolls away.
//
// The videos loop forever, so this is autoplay without the cost of it: an `autoplay`
// attribute makes the browser fetch each clip at page load, and the real-world and book
// clips alone are tens of megabytes. Here they stay at preload="metadata" until they are
// about to come into view, and only the clip you are looking at is decoding.
function setupVideoAutoplay() {
    const videos = document.querySelectorAll('main video');
    if (videos.length === 0) return;

    if (!('IntersectionObserver' in window)) {
        // No observer: fall back to playing everything and let the browser sort it out.
        videos.forEach(video => video.play().catch(() => {}));
        return;
    }

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const video = entry.target;
            if (entry.isIntersecting) {
                if (video.preload !== 'auto') video.preload = 'auto';
                // Autoplay is only allowed while muted, and a user may have unmuted this one.
                video.play().catch(() => {});
            } else {
                video.pause();
            }
        });
    }, {
        threshold: 0.25,      // a quarter of the video is enough to count as "watching"
        rootMargin: '200px'   // start fetching just before it reaches the viewport
    });

    videos.forEach(video => {
        video.loop = true;
        video.muted = true;   // required for autoplay to be permitted at all
        video.playsInline = true;
        observer.observe(video);
    });
}

// Start the videos on their own, not from inside the jQuery block below. That block depends
// on jQuery (a CDN script) and on bulmaCarousel, and if either is missing it throws before it
// reaches the videos -- which is exactly how autoplay silently stopped working.
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupVideoAutoplay);
} else {
    setupVideoAutoplay();
}

// Carousel and slider are template leftovers: this page has no .carousel or .slider element.
// Keep them, but never let them take the rest of the script down with them.
if (typeof $ !== 'undefined') {
    $(document).ready(function() {
        var options = {
            slidesToScroll: 1,
            slidesToShow: 1,
            loop: true,
            infinite: true,
            autoplay: true,
            autoplaySpeed: 5000,
        };

        try {
            if (typeof bulmaCarousel !== 'undefined') bulmaCarousel.attach('.carousel', options);
            if (typeof bulmaSlider !== 'undefined') bulmaSlider.attach();
        } catch (e) {
            console.warn('carousel/slider init skipped:', e);
        }
    });
}
