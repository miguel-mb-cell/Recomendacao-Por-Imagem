document.querySelectorAll('.image-container').forEach(container => {
    container.addEventListener('click', function () {
        const imagePath = this.querySelector('img').getAttribute('src');

        const modalBody = document.querySelector('.modal-body');
        const images = modalBody.querySelectorAll('img');
        images.forEach(img => img.remove());

        const loader = document.getElementById("loader");

        loader.style.display = "block";
        
        fetch('http://127.0.0.1:5000/get_similar', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image_path: imagePath }),
        })
            .then(response => response.json())
            .then(data => {
                loader.style.display = "none";

                if (data.error) {
                    console.error(data.error);
                    return;
                }

                const similarImages = data.similar_images;

                similarImages.forEach(imgPath => {
                    const imgElement = document.createElement('img');
                    imgElement.src = imgPath;
                    modalBody.appendChild(imgElement);
                });

                const modal = new bootstrap.Modal(document.getElementById('imageModal'));
                modal.show();
            })
            .catch(error => {
                loader.style.display = "none";
                console.error('Error:', error)
            });
    });
});