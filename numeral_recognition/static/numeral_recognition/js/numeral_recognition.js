function csrfSafeMethod(method) {
    // these HTTP methods do not require CSRF protection
    return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
}

$( function() {
    $("form").submit( function(event) {
        event.preventDefault();
        var form = $(this);
        var fd = new FormData();

        //CSRF対策用トークンを取得
        var csrftoken = Cookies.get('csrftoken');

        var img = $("input[name='img']").prop('files')[0];

        // タグ名はform.pyで指定した変数名を使用しないと正常にバリデーションされない
        fd.append('img', img);

        $.ajax({
                url: form.prop('action'),
                method: form.prop('method'),
                data: fd,
                timeout: 10000,
                dataType: 'json',
                // 送信データの最初にCSRF対策用トークンを追加
                beforeSend: function(xhr, settings) {
                    if (!csrfSafeMethod(settings.type) && !this.crossDomain) {
                        xhr.setRequestHeader('X-CSRFToken', csrftoken);
                    }
                },
                processData: false,
                contentType: false,
        }).done( function(response) {
            if (response.error_msg) {
                $('#errorMsg').text('Error: ' + response.error_msg);
            } else {
                $('#result').text(response.result);
            }

        })
    });
});