function csrfSafeMethod(method) {
    // these HTTP methods do not require CSRF protection
    return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
}

// 参考: https://www.otwo.jp/blog/canvas-drawing/
class Canvas {
    constructor() {
        this.cv =  $("#nr_canvas")[0]
        this.ct = this.cv.getContext('2d');

        // クリック中の判定(1:クリック開始, 2:クリック中)
        this.clickFlg = 0;

        // 背景色
        this.bgColor = "rgb(255,255,255)";

        // 線の色
        this.strokeStyle = "255, 255, 255, 1";

        // 線の幅
        this.lineWidth = 15;

        // Canvasの幅・高さ
        this.cnvWidth = this.cv.width;
        this.cnvHeight = this.cv.height;

        this.setBgColor()
        this.setLineStyle()

    }

    setBgColor() {
        this.ct.fillStyle = this.bgColor;
        this.ct.fillRect(0, 0, this.cnvWidth, this.cnvHeight);
    }

    setLineStyle() {
        this.ct.lineWidth = this.lineWidth;
        this.ct.strokeStyle = this.strokeStyle;
    }

    draw(x, y) {
        if (this.clickFlg == "1") {
            this.clickFlg = 2;
            this.ct.beginPath();
            this.ct.lineCap = "round";
            this.ct.moveTo(x, y);
        } else {
            this.ct.lineTo(x, y);
        }
        this.ct.stroke();
        }

    clear() {
      this.ct.clearRect(0, 0, this.cnvWidth, this.cnvHeight);
      this.setBgColor();
    }
}

// Canvas画像送信処理
function sendCanvasImg() {
    //CSRF対策用トークンを取得
    var csrftoken = Cookies.get('csrftoken');

    var img = $("#nr_canvas")[0].toDataURL("image/png");

    var fd = new FormData();

    fd.append('img', img);

   $.ajax({
        url: "/numeralRecognition/a/",
        method: "post",
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
            $('#canvasErrorMsg').text('Error: ' + response.error_msg);
        } else {
            $('#canvasResult').text(response.result);
        }

    })
}

$( function() {
    // NumeralRecognition用キャンバス処理
    const canvas = new Canvas()
    $("#nr_canvas").mousedown(function(){
        // マウス押下開始
        canvas.clickFlg = 1;
    }).mouseup(function(){
        //　マウス押下終了
        canvas.clickFlg = 0;
        sendCanvasImg();
    }).mousemove(e => {
        // マウス移動処理
        if(!canvas.clickFlg) return false;
        canvas.draw(e.offsetX, e.offsetY);
    });

    $("#canvasClear").click( function(event) {
        canvas.clear();
    })

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
                $('#imageErrorMsg').text('Error: ' + response.error_msg);
            } else {
                $('#imageResult').text(response.result);
            }

        })
    });
});