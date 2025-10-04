document.addEventListener('DOMContentLoaded', () => {
    // Maneja el menú desplegable de "Cuenta"
    const dropdownButton = document.querySelector('.dropdown-button');
    const dropdownMenu = document.querySelector('.dropdown-menu');

    dropdownButton.addEventListener('click', () => {
        dropdownMenu.parentElement.classList.toggle('active');
    });

    // Oculta el menú si se hace clic fuera de él
    window.addEventListener('click', (e) => {
        if (!dropdownButton.contains(e.target) && !dropdownMenu.contains(e.target)) {
            dropdownMenu.parentElement.classList.remove('active');
        }
    });

    // Código para desplazamiento suave (smooth scrolling)
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });
});