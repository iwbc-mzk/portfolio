$( function() {
    $("a.nav-link, a.navbar-brand").click( function () {
        var hrefValue = jQuery(this).attr( 'href' );
        $(hrefValue).animatescroll({padding:55});

        $(this).addClass('active');
    });

    window.addEventListener( 'scroll', () => {
        var scrollTop = $(window).scrollTop();
        if(scrollTop > 70){
            $('#navbarTop').removeClass('navbar-dark').addClass('navbar-light').addClass('menu-shadow').addClass('menu-scrolled', 600);
        }else{
            $('#navbarTop').removeClass('navbar-light').addClass('navbar-dark').removeClass('menu-shadow').removeClass('menu-scrolled', 600);
        }
    }, false);
})
