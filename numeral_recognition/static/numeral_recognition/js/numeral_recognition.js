function csrfSafeMethod(method) {
    // these HTTP methods do not require CSRF protection
    return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
}

// クリック中の判定(1:クリック開始, 2:クリック中)
var clickFlg = 0;

// 参考: https://www.otwo.jp/blog/canvas-drawing/
function draw(x, y) {
    var cv = $("#nr_canvas")[0]
    var ct = cv.getContext('2d');

    // 線の色
    var strokeStyle = "255, 255, 255, 1";
    // 線の太さ
    var lineWidth = 5;

    ct.lineWidth = lineWidth;
    ct.strokeStyle = strokeStyle;

    if (clickFlg == "1") {
        clickFlg = 2;
        ct.beginPath();
        ct.lineCap = "round";
        ct.moveTo(x, y);
    } else {
        ct.lineTo(x, y);
    }
    ct.stroke();
}

$( function() {
    // NumeralRecognition用キャンバス処理
    $("#nr_canvas").mousedown(function(){
        // マウス押下開始
        clickFlg = 1;
    }).mouseup(function(){
        //　マウス押下終了
        clickFlg = 0;
    }).mousemove(e => {
        // マウス移動処理
        if(!clickFlg) return false;
        draw(e.offsetX, e.offsetY);
    });

    // 手書き画像送信フォーム用処理
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