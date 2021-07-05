$(document).ready(function(){
    $('.uploadline').change(function () {
        var form = document.forms['form1'];
        form.submit();
        return true;
    });
});

function initColor(pointData, width, height, type) {
    let embed = 0;
    if (type == "point") {
        embed = 255;
    }
    for (var i = 0; i < height; i++) {
        for (var j = 0; j < width; j++) {
            var idx = (j + i * width) * 4;
            pointData[idx] = embed;
            pointData[idx + 1] = embed;
            pointData[idx + 2] = embed;
            pointData[idx + 3] = 255;
        }
    }

    return pointData
}

window.addEventListener('load', () => {
    const canvas = document.querySelector('#canvas_line');
    const srcid = document.getElementById('canvas_line').getAttribute('src');
    const height = document.getElementById('canvas_line').getAttribute('height');
    const width = document.getElementById('canvas_line').getAttribute('width');
    const context = canvas.getContext('2d');

    const pointCanvas = document.querySelector('#canvas_points');
    const placeCanvas = document.querySelector('#canvas_places');
    const pointContext = pointCanvas.getContext('2d');
    const placeContext = placeCanvas.getContext('2d');

    var pointData = pointContext.createImageData(width, height);
    var placeData = placeContext.createImageData(width, height);

    pointData.data = initColor(pointData.data, width, height, "point");
    placeData.data = initColor(placeData.data, width, height);

    pointContext.putImageData(pointData, 0, 0);
    placeContext.putImageData(placeData, 0, 0);

    let currentColor = '#FEDCBD';
    let lineWidth = 5;

    function draw(event) {
        let x = event.offsetX;
        let y = event.offsetY;
        context.fillStyle = currentColor;
        context.fillRect(x-lineWidth, y-lineWidth, lineWidth*2, lineWidth*2);

        pointContext.fillStyle = currentColor;
        pointContext.fillRect(x-lineWidth, y-lineWidth, lineWidth*2, lineWidth*2);

        placeContext.fillStyle = "#FFFFFF";
        placeContext.fillRect(x-lineWidth, y-lineWidth, lineWidth*2, lineWidth*2);
    }

    function clear(event) {
        context.clearRect(0, 0, canvas.width, canvas.height);
        reloadImage();
    }

    function changeAccurate(event) {
        lineWidth = 3;
    }

    function changeRough(event) {
        lineWidth = 5;
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
        canvas.addEventListener('mousedown', draw);
    }

    initEventHandler();

    $(document).on('click', '.palette', function() {currentColor = $(this).attr('id');});
    $(document).on('click', '.allclear', clear);
    $(document).on('click', '.accurate', changeAccurate);
    $(document).on('click', ".rough", changeRough);

    document.querySelector('#colorize').addEventListener('click', (e)=>{
        e.preventDefault();
    });
});

function pointColorize(event) {
    const uri = document.querySelector('#canvas_line').toDataURL('image/png');
    const point = document.querySelector('#canvas_points').toDataURL('image/png');
    const place = document.querySelector('#canvas_places').toDataURL('image/png');
    const filename = document.getElementById('canvas_line').getAttribute('src');
    const param  = {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json; charset=utf-8'
        },
        body: JSON.stringify({image: uri, point: point, place: place, name: filename})
    };

    fetch('/point', param)
        .then((res)=>{
            res.json().then(function(data) {
                var y_result = document.querySelector('.y_result');
                y_result.src = 'data:image/png;base64,' + data['data'];
                y_result.width = data['w'];
                y_result.height = data['h'];
            });
        });
}

document.querySelector('.btn_flat').addEventListener('click', pointColorize);
