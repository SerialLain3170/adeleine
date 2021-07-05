$(document).ready(function(){
    $('.uploadline').change(function () {
        var form = document.forms['form1'];
        form.submit();
        return true;
    });
});

window.addEventListener('load', () => {
    const canvas = document.querySelector('#canvas_line');
    const srcid = document.getElementById('canvas_line').getAttribute('src');
    console.log(srcid);
    const context = canvas.getContext('2d');

    const lastPosition = { x: null, y: null };
    let isDrag = false;
    let currentColor = '#FEDCBD';

    function draw(x, y) {
        if(!isDrag) {
        return;
        }
        context.lineCap = 'round';
        context.lineJoin = 'round';
        context.lineWidth = 5;
        context.strokeStyle = currentColor;
        if (lastPosition.x === null || lastPosition.y === null) {
            context.moveTo(x, y);
        } else {
            context.moveTo(lastPosition.x, lastPosition.y);
        }
        context.lineTo(x, y);
        context.stroke();

        lastPosition.x = x;
        lastPosition.y = y;
    }

    function clear() {
        context.clearRect(0, 0, canvas.width, canvas.height);
        reloadImage();
    }

    function dragStart(event) {
        context.beginPath();

        isDrag = true;
    }

    function dragEnd(event) {
        context.closePath();
        isDrag = false;
        lastPosition.x = null;
        lastPosition.y = null;
    }

    function loadImage() {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                context.drawImage(img, 0, 0)
                resolve(img);
            }
            img.src = srcid;
        });
    }

    function reloadImage() {
        const img = new Image();
        img.onload = () => {
            context.drawImage(img, 0, 0)
            resolve(img);
        }
        img.src = srcid;
    }

    async function initEventHandler() {
        //const clearButton = document.querySelector('#clear-button');
        //clearButton.addEventListener('click', clear);
    
        // 消しゴムモードを選択したときの挙動
        //const eraserButton = document.querySelector('#eraser-button');
        //eraserButton.addEventListener('click', () => {
          // 消しゴムと同等の機能を実装したい場合は現在選択している線の色を
          // 白(#FFFFFF)に変更するだけでよい
        //  currentColor = '#FFFFFF';
        //});
        await loadImage();
        canvas.addEventListener('mousedown', dragStart);
        canvas.addEventListener('mouseup', dragEnd);
        canvas.addEventListener('mouseout', dragEnd);
        canvas.addEventListener('mousemove', (event) => {
            draw(event.offsetX, event.offsetY);
        });
    }

    //function initColorPalette() {
    //   const joe = colorjoe.rgb('color-palette', currentColor);

    //    joe.on('done', color => {
    //        currentColor = color.hex();
    //    });
    // }

    initEventHandler();

    $(document).on('click', '.palette', function() {
        currentColor = $(this).attr('id');
        console.log(currentColor);
    });

    $(document).on('click', '.allclear', function() {
        clear();
    });

    document.querySelector('#colorize').addEventListener('click', (e)=>{
        e.preventDefault();
    });
});

document.querySelector('.btn_scribble').addEventListener('click', ()=>{
    const uri = document.querySelector('#canvas_line').toDataURL('image/png');
    const filename = document.getElementById('canvas_line').getAttribute('src');
    const param  = {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json; charset=utf-8'
        },
        body: JSON.stringify({data: uri, name: filename})
    };
    sendServer('/scribble', param);
});

function sendServer(url, param){
    fetch(url, param)
        .then((res)=>{
            res.json().then(function(data) {
                var y_result = document.querySelector('.y_result');
                y_result.src = 'data:image/png;base64,' + data['data'];
                y_result.width = data['w'];
                y_result.height = data['h'];
            });
        });
}
