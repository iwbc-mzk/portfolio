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

        var aboutPos = $('#about').offset().top;
        if(scrollTop > aboutPos - windowHeight + windowHeight/5){
            $('#about').removeClass('fade-off', 700);
        }else{
            $('#about').addClass('fade-off', 700);
        }

        var skillsPos = $('#skills').offset().top;
        if(scrollTop > skillsPos - windowHeight + windowHeight/5){
            $('#skills').removeClass('fade-off', 700);
        }else{
            $('#skills').addClass('fade-off', 700);
        }

        var workExperiencePos = $('#workExperience').offset().top;
        if(scrollTop > workExperiencePos - windowHeight + windowHeight/5){
            $('#workExperience').removeClass('fade-off', 700);
        }else{
            $('#workExperience').addClass('fade-off', 700);
        }

        var worksPos = $('#works').offset().top;
        if(scrollTop > worksPos - windowHeight + windowHeight/5){
            $('#works').removeClass('fade-off', 700);
        }else{
            $('#works').addClass('fade-off', 700);
        }
    })
})
