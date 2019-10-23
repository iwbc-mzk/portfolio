$( function() {
    $("a.nav-link, a.navbar-brand").click( function () {
        var hrefValue = $(this).attr('href');
        $(hrefValue).animatescroll({scrollSpeed:1000, padding:55}, 1000);

        $(this).addClass('active');

        $('.navbar-toggler').attr('aria-expanded', 'false').addClass('collapsed');
        $('.navbar-collapse').collapse('hide');
        return false;
    });

    $(window).on("scroll", function(){
        var scrollTop = $(window).scrollTop();
        var windowHeight = $(window).height();
        if(scrollTop > 70){
            $('#navbarTop').removeClass('navbar-dark').addClass('navbar-light').addClass('menu-shadow').addClass('menu-scrolled', 600);
        }else{
            $('#navbarTop').removeClass('navbar-light').addClass('navbar-dark').removeClass('menu-shadow').removeClass('menu-scrolled', 600);
        }
    })
})
