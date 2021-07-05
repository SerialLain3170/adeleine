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
    const context = canvas.getContext('2d');

    const lastPosition = { x: null, y: null };
    let isDrag = false;
    let currentColor = '#FEDCBD';

    function draw(event) {
        if(!isDrag) {
            return;
        }
        let x = event.offsetX;
        let y = event.offsetY;
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

    function clear(event) {
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
        await loadImage();
        canvas.addEventListener('mousedown', dragStart);
        canvas.addEventListener('mouseup', dragEnd);
        canvas.addEventListener('mouseout', dragEnd);
        canvas.addEventListener('mousemove', draw);
        document.querySelector('#colorize').addEventListener('click', (e)=>{
            e.preventDefault();
        });
    }

    initEventHandler();

    $(document).on('click', '.palette', function() {currentColor = $(this).attr('id');});
    $(document).on('click', '.allclear', clear);
});

function flatColorize(event) {
    const uri = document.querySelector('#canvas_line').toDataURL('image/png');
    const filename = document.getElementById('canvas_line').getAttribute('src');
    const param  = {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json; charset=utf-8'
        },
        body: JSON.stringify({image: uri, name: filename})
    };
    fetch('/flatten', param)
        .then((res)=>{
            res.json().then(function(data) {
                var element = document.querySelector('.y_result');
                element.src = 'data:image/png;base64,' + data['data'];
                element.width = data['w'];
                element.height = data['h'];
            });
        });
}

document.querySelector('.btn_flat').addEventListener('click', flatColorize)
