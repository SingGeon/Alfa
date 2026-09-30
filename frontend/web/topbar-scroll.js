(function () {
  var topbar = document.querySelector('.topbar');
  if (!topbar) return;

  var lastY = window.scrollY;
  var ticking = false;
  var revealThreshold = 80;

  function onScroll() {
    var y = window.scrollY;
    if (y > lastY && y > revealThreshold) {
      topbar.classList.add('topbar--hidden');
    } else {
      topbar.classList.remove('topbar--hidden');
    }
    lastY = y;
    ticking = false;
  }

  window.addEventListener('scroll', function () {
    if (!ticking) {
      window.requestAnimationFrame(onScroll);
      ticking = true;
    }
  }, { passive: true });
})();
