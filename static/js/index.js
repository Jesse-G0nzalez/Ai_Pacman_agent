window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}

$(document).ready(function() {
  // Navbar burger toggle
  $(".navbar-burger").click(function() {
    $(".navbar-burger").toggleClass("is-active");
    $(".navbar-menu").toggleClass("is-active");
  });

  // Team‐members carousel: show 3 at a time
  var teamOptions = {
    slidesToScroll:  1,
    slidesToShow:    3,
    loop:            true,
    infinite:        true,
    navigation:      true,
    pagination:      false,
    gap:             24,
    breakpoints: {
      768: { slidesToShow: 1, gap: 16 }
    }
  };
  bulmaCarousel.attach('#results-carousel', teamOptions);

  // Charts carousel: show 1 at a time
  var chartOptions = {
    slidesToScroll:  1,
    slidesToShow:    1,
    loop:            true,
    infinite:        true,
    navigation:      true,
    pagination:      false,
    gap:             16
  };
  bulmaCarousel.attach('#charts-carousel', chartOptions);

  // Interpolation slider setup
  preloadInterpolationImages();
  $('#interpolation-slider').on('input', function() {
    setInterpolationImage(this.value);
  });
  setInterpolationImage(0);
  $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

  // Bulma slider (if you have any .slider elements)
  bulmaSlider.attach();
});
