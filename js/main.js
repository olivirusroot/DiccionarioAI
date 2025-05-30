document.addEventListener('DOMContentLoaded', function() {
  // Referencias a elementos del DOM
  const searchInput = document.getElementById('search-input');
  const termsGrid = document.getElementById('terms-grid');
  const alphabetIndex = document.getElementById('alphabet-index');
  const resultsCount = document.getElementById('results-count');
  const noResults = document.getElementById('no-results');
  const resetFilters = document.getElementById('reset-filters');
  const loader = document.getElementById('loader');
  const termModal = document.getElementById('term-modal');
  const modalClose = document.getElementById('modal-close');
  const modalTitle = document.getElementById('modal-title');
  const modalCategory = document.getElementById('modal-category');
  const modalComplexity = document.getElementById('modal-complexity');
  const modalDefinition = document.getElementById('modal-definition');
  const relatedTermsList = document.getElementById('related-terms-list');
  const backToTop = document.getElementById('back-to-top');
  const themeToggle = document.getElementById('theme-toggle');
  const notification = document.getElementById('notification');
  const notificationMessage = document.getElementById('notification-message');
  
  // Estado de la aplicación
  let currentFilter = 'all';
  let currentComplexity = 'all';
  let currentSort = 'alphabetical';
  let currentLetter = 'all';
  let currentSearch = '';
  let filteredTerms = [];
  let isDarkMode = true;
  
  // Inicialización
  init();
  
  // Función de inicialización
  function init() {
    // Mostrar loader
    showLoader();
    
    // Inicializar filtros
    setupFilters();
    
    // Inicializar índice alfabético
    setupAlphabetIndex();
    
    // Inicializar búsqueda
    setupSearch();
    
    // Inicializar modal
    setupModal();
    
    // Inicializar botón volver arriba
    setupBackToTop();
    
    // Inicializar tema
    setupTheme();
    
    // Cargar términos
    loadTerms();
  }
  
  // Configuración de filtros
  function setupFilters() {
    // Filtros de categoría
    document.querySelectorAll('.filter-btn[data-filter]').forEach(btn => {
      btn.addEventListener('click', function() {
        document.querySelectorAll('.filter-btn[data-filter]').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
        currentFilter = this.dataset.filter;
        filterAndRenderTerms();
      });
    });
    
    // Filtros de complejidad
    document.querySelectorAll('.filter-btn[data-complexity]').forEach(btn => {
      btn.addEventListener('click', function() {
        document.querySelectorAll('.filter-btn[data-complexity]').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
        currentComplexity = this.dataset.complexity;
        filterAndRenderTerms();
      });
    });
    
    // Ordenación
    document.querySelectorAll('.filter-btn[data-sort]').forEach(btn => {
      btn.addEventListener('click', function() {
        document.querySelectorAll('.filter-btn[data-sort]').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
        currentSort = this.dataset.sort;
        filterAndRenderTerms();
      });
    });
    
    // Botón de reset
    resetFilters.addEventListener('click', function() {
      resetAllFilters();
    });
  }
  
  // Configuración del índice alfabético
  function setupAlphabetIndex() {
    const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
    
    // Botón "Todos"
    const allBtn = document.createElement('button');
    allBtn.className = 'alphabet-btn active';
    allBtn.textContent = '#';
    allBtn.dataset.letter = 'all';
    allBtn.addEventListener('click', function() {
      setActiveLetter('all');
    });
    alphabetIndex.appendChild(allBtn);
    
    // Botones para cada letra
    alphabet.forEach(letter => {
      const btn = document.createElement('button');
      btn.className = 'alphabet-btn';
      btn.textContent = letter;
      btn.dataset.letter = letter.toLowerCase();
      btn.addEventListener('click', function() {
        setActiveLetter(letter.toLowerCase());
      });
      alphabetIndex.appendChild(btn);
    });
  }
  
  // Configuración de la búsqueda
  function setupSearch() {
    searchInput.addEventListener('input', function() {
      currentSearch = this.value.trim().toLowerCase();
      filterAndRenderTerms();
    });
    
    // Limpiar búsqueda con Escape
    searchInput.addEventListener('keydown', function(e) {
      if (e.key === 'Escape') {
        this.value = '';
        currentSearch = '';
        filterAndRenderTerms();
      }
    });
  }
  
  // Configuración del modal
  function setupModal() {
    // Cerrar modal
    modalClose.addEventListener('click', closeModal);
    
    // Cerrar modal con Escape
    document.addEventListener('keydown', function(e) {
      if (e.key === 'Escape' && termModal.classList.contains('active')) {
        closeModal();
      }
    });
    
    // Cerrar modal al hacer clic fuera del contenido
    termModal.addEventListener('click', function(e) {
      if (e.target === termModal) {
        closeModal();
      }
    });
  }
  
  // Configuración del botón volver arriba
  function setupBackToTop() {
    // Mostrar/ocultar botón según scroll
    window.addEventListener('scroll', function() {
      if (window.scrollY > 300) {
        backToTop.classList.add('visible');
      } else {
        backToTop.classList.remove('visible');
      }
    });
    
    // Acción del botón
    backToTop.addEventListener('click', function() {
      window.scrollTo({
        top: 0,
        behavior: 'smooth'
      });
    });
  }
  
  // Configuración del tema
  function setupTheme() {
    themeToggle.addEventListener('click', function() {
      toggleTheme();
    });
  }
  
  // Cargar términos
  function loadTerms() {
    // Simular carga
    setTimeout(() => {
      filteredTerms = [...dictionaryData];
      filterAndRenderTerms();
      hideLoader();
    }, 500);
  }
  
  // Filtrar y renderizar términos
  function filterAndRenderTerms() {
    // Aplicar filtros
    filteredTerms = dictionaryData.filter(term => {
      // Filtro de categoría
      const categoryMatch = currentFilter === 'all' || term.category === currentFilter;
      
      // Filtro de complejidad
      const complexityMatch = currentComplexity === 'all' || term.complexity === parseInt(currentComplexity);
      
      // Filtro de letra
      const letterMatch = currentLetter === 'all' || term.term.toLowerCase().startsWith(currentLetter);
      
      // Filtro de búsqueda
      const searchMatch = currentSearch === '' || 
        term.term.toLowerCase().includes(currentSearch) || 
        term.shortDefinition.toLowerCase().includes(currentSearch) ||
        term.fullDefinition.toLowerCase().includes(currentSearch) ||
        term.tags.some(tag => tag.toLowerCase().includes(currentSearch));
      
      return categoryMatch && complexityMatch && letterMatch && searchMatch;
    });
    
    // Ordenar términos
    sortTerms();
    
    // Actualizar contador de resultados
    updateResultsCount();
    
    // Renderizar términos
    renderTerms();
  }
  
  // Ordenar términos
  function sortTerms() {
    switch (currentSort) {
      case 'alphabetical':
        filteredTerms.sort((a, b) => a.term.localeCompare(b.term));
        break;
      case 'complexity':
        filteredTerms.sort((a, b) => a.complexity - b.complexity);
        break;
    }
  }
  
  // Actualizar contador de resultados
  function updateResultsCount() {
    resultsCount.textContent = filteredTerms.length;
    
    if (filteredTerms.length === 0) {
      noResults.style.display = 'block';
    } else {
      noResults.style.display = 'none';
    }
  }
  
  // Renderizar términos
  function renderTerms() {
    // Limpiar grid
    termsGrid.innerHTML = '';
    
    // Renderizar cada término
    filteredTerms.forEach(term => {
      const termCard = createTermCard(term);
      termsGrid.appendChild(termCard);
    });
  }
  
  // Crear tarjeta de término
  function createTermCard(term) {
    const termCard = document.createElement('div');
    termCard.className = 'term-card animate-fade-in';
    termCard.dataset.id = term.id;
    
    // Encontrar categoría
    const category = categories.find(cat => cat.id === term.category);
    
    // Encontrar nivel de complejidad
    const complexity = complexityLevels.find(comp => comp.level === term.complexity);
    
    // Contenido de la tarjeta
    termCard.innerHTML = `
      <div class="term-header">
        <h3 class="term-title">${highlightSearchTerm(term.term)}</h3>
        <span class="category-btn ${term.category}" style="background-color: ${category.color}30; color: ${category.color}; border: 1px solid ${category.color}50;">
          ${category.name}
        </span>
      </div>
      
      <p class="term-definition">${highlightSearchTerm(term.shortDefinition)}</p>
      
      <div class="term-footer">
        <div class="complexity-indicator">
          ${createComplexityDots(term.complexity)}
          <span class="complexity-label">${complexity.name}</span>
        </div>
        
        <span class="term-more">
          Ver más
          <i class="fas fa-chevron-right"></i>
        </span>
      </div>
    `;
    
    // Evento de clic
    termCard.addEventListener('click', function() {
      openTermDetail(term);
    });
    
    return termCard;
  }
  
  // Crear indicadores de complejidad (puntos)
  function createComplexityDots(level) {
    let dots = '';
    
    for (let i = 1; i <= 3; i++) {
      const isActive = i <= level;
      const complexityClass = level === 1 ? 'basic' : level === 2 ? 'intermediate' : 'advanced';
      dots += `<div class="complexity-dot ${isActive ? 'active ' + complexityClass : ''}"></div>`;
    }
    
    return dots;
  }
  
  // Resaltar término de búsqueda
  function highlightSearchTerm(text) {
    if (!currentSearch) return text;
    
    const regex = new RegExp(`(${escapeRegExp(currentSearch)})`, 'gi');
    return text.replace(regex, '<span class="highlight">$1</span>');
  }
  
  // Escapar caracteres especiales para RegExp
  function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }
  
  // Abrir detalle de término
  function openTermDetail(term) {
    // Encontrar categoría
    const category = categories.find(cat => cat.id === term.category);
    
    // Encontrar nivel de complejidad
    const complexity = complexityLevels.find(comp => comp.level === term.complexity);
    
    // Actualizar contenido del modal
    modalTitle.textContent = term.term;
    modalCategory.textContent = category.name;
    modalCategory.className = `category-btn ${term.category}`;
    modalCategory.style.backgroundColor = `${category.color}30`;
    modalCategory.style.color = category.color;
    modalCategory.style.border = `1px solid ${category.color}50`;
    
    // Actualizar complejidad
    const complexityDots = document.querySelectorAll('.modal-meta .complexity-dot');
    complexityDots.forEach((dot, index) => {
      dot.className = 'complexity-dot';
      if (index < term.complexity) {
        dot.classList.add('active');
        dot.classList.add(term.complexity === 1 ? 'basic' : term.complexity === 2 ? 'intermediate' : 'advanced');
      }
    });
    modalComplexity.textContent = complexity.name;
    
    // Actualizar definición
    modalDefinition.textContent = term.fullDefinition;
    
    // Actualizar términos relacionados
    relatedTermsList.innerHTML = '';
    
    if (term.relatedTerms && term.relatedTerms.length > 0) {
      document.getElementById('related-terms').style.display = 'block';
      
      term.relatedTerms.forEach(relatedTermId => {
        const relatedTerm = dictionaryData.find(t => t.id === relatedTermId);
        
        if (relatedTerm) {
          const link = document.createElement('a');
          link.href = '#';
          link.className = 'related-term-link';
          link.textContent = relatedTerm.term;
          link.addEventListener('click', function(e) {
            e.preventDefault();
            openTermDetail(relatedTerm);
          });
          
          relatedTermsList.appendChild(link);
        }
      });
    } else {
      document.getElementById('related-terms').style.display = 'none';
    }
    
    // Mostrar modal
    termModal.classList.add('active');
    
    // Deshabilitar scroll del body
    document.body.style.overflow = 'hidden';
  }
  
  // Cerrar modal
  function closeModal() {
    termModal.classList.remove('active');
    document.body.style.overflow = '';
  }
  
  // Establecer letra activa
  function setActiveLetter(letter) {
    // Actualizar estado
    currentLetter = letter;
    
    // Actualizar UI
    document.querySelectorAll('.alphabet-btn').forEach(btn => {
      btn.classList.remove('active');
      if (btn.dataset.letter === letter) {
        btn.classList.add('active');
      }
    });
    
    // Filtrar términos
    filterAndRenderTerms();
    
    // Scroll al inicio
    window.scrollTo({
      top: document.querySelector('.terms-grid').offsetTop - 100,
      behavior: 'smooth'
    });
  }
  
  // Resetear todos los filtros
  function resetAllFilters() {
    // Resetear estado
    currentFilter = 'all';
    currentComplexity = 'all';
    currentLetter = 'all';
    currentSearch = '';
    
    // Resetear UI
    document.querySelectorAll('.filter-btn[data-filter]').forEach(btn => {
      btn.classList.remove('active');
      if (btn.dataset.filter === 'all') {
        btn.classList.add('active');
      }
    });
    
    document.querySelectorAll('.filter-btn[data-complexity]').forEach(btn => {
      btn.classList.remove('active');
      if (btn.dataset.complexity === 'all') {
        btn.classList.add('active');
      }
    });
    
    document.querySelectorAll('.alphabet-btn').forEach(btn => {
      btn.classList.remove('active');
      if (btn.dataset.letter === 'all') {
        btn.classList.add('active');
      }
    });
    
    searchInput.value = '';
    
    // Filtrar términos
    filterAndRenderTerms();
    
    // Mostrar notificación
    showNotification('Filtros restablecidos', 'success');
  }
  
  // Mostrar loader
  function showLoader() {
    loader.style.display = 'flex';
    termsGrid.style.opacity = '0.5';
  }
  
  // Ocultar loader
  function hideLoader() {
    loader.style.display = 'none';
    termsGrid.style.opacity = '1';
  }
  
  // Mostrar notificación
  function showNotification(message, type = '') {
    notificationMessage.textContent = message;
    notification.className = `notification ${type}`;
    notification.classList.add('show');
    
    setTimeout(() => {
      notification.classList.remove('show');
    }, 3000);
  }
  
  // Cambiar tema
  function toggleTheme() {
    isDarkMode = !isDarkMode;
    
    if (isDarkMode) {
      document.documentElement.style.setProperty('--primary-bg', '#0a1933');
      document.documentElement.style.setProperty('--secondary-bg', '#0c2045');
      document.documentElement.style.setProperty('--text-primary', '#ffffff');
      document.documentElement.style.setProperty('--text-secondary', '#e0e0e0');
      themeToggle.innerHTML = '<i class="fas fa-moon"></i>';
    } else {
      document.documentElement.style.setProperty('--primary-bg', '#f8f9fa');
      document.documentElement.style.setProperty('--secondary-bg', '#e9ecef');
      document.documentElement.style.setProperty('--text-primary', '#212529');
      document.documentElement.style.setProperty('--text-secondary', '#495057');
      themeToggle.innerHTML = '<i class="fas fa-sun"></i>';
    }
    
    // Guardar preferencia
    localStorage.setItem('darkMode', isDarkMode);
    
    // Mostrar notificación
    showNotification(`Modo ${isDarkMode ? 'oscuro' : 'claro'} activado`);
  }
});
