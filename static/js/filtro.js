document.querySelectorAll('.filter-link').forEach(link => {
    link.addEventListener('click', function (event) {
        event.preventDefault();
        const filterParam = this.getAttribute('data-filter');
        const baseUrl = window.location.href.split('?')[0];
        const newUrl = `${baseUrl}?${filterParam}&page=1`;
        window.location.href = newUrl;
    });
});

document.addEventListener('DOMContentLoaded', function () {
    const urlParams = new URLSearchParams(window.location.search);
    const apparelParam = urlParams.get('apparel');
    const footwearParam = urlParams.get('footwear');

    const roupasButton = document.getElementById('roupasDropdown');
    const sapatosButton = document.getElementById('sapatosDropdown');

    if (apparelParam) {
        roupasButton.classList.remove('btn-secondary');
        roupasButton.classList.add('btn-primary');
        sapatosButton.classList.add('btn-secondary');
        sapatosButton.classList.remove('btn-primary');
    }

    if (footwearParam) {
        sapatosButton.classList.remove('btn-secondary');
        sapatosButton.classList.add('btn-primary');
        roupasButton.classList.add('btn-secondary');
        roupasButton.classList.remove('btn-primary');
    }
});